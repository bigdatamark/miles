# [MODULE DOC] Unit tests for the structured logging helpers: the value formatter and the logfmt
# line builder (both private), the size-capping pruner, the log_structured emitter, and the
# with_logs decorator for sync and async methods. Log output is captured with the pytest log
# capture fixture; a small fake actor class defined here is decorated to exercise the decorator.
import asyncio
import logging

import pytest

from miles.utils.tracking_utils.structured_log import (
    _format_value,
    _to_logfmt,
    log_structured,
    prune_for_log,
    with_logs,
)


class TestFormatValue:
    # [CLASS DOC] Tests of the single-value formatter: numbers, booleans, none, strings (quoting
    # and escaping), sequences (comma joined) and mappings (compact quoted JSON).
    def test_int_renders_bare(self):
        # [FUNCTION DOC] Asserts the integer 7 renders as the bare text 7.
        """An int renders as a bare number."""
        assert _format_value(7) == "7"

    def test_float_renders_bare(self):
        # [FUNCTION DOC] Asserts the float 49.3 renders as the bare text 49.3.
        """A float renders as a bare number."""
        assert _format_value(49.3) == "49.3"

    def test_true_and_false_render_lowercase(self):
        # [FUNCTION DOC] Asserts true and false booleans render as lowercase true and false.
        """Bools render as lowercase true/false (and take precedence over the int branch)."""
        assert _format_value(True) == "true"
        assert _format_value(False) == "false"

    def test_none_renders_as_empty(self):
        # [FUNCTION DOC] Asserts none renders as the empty string.
        """None renders as an empty value."""
        assert _format_value(None) == ""

    def test_empty_string_renders_as_empty(self):
        # [FUNCTION DOC] Asserts an empty string renders as the empty string with no quotes.
        """An empty string renders as an empty value with no quotes."""
        assert _format_value("") == ""

    def test_plain_string_renders_bare(self):
        # [FUNCTION DOC] Asserts a string without space, equals sign or quote renders unchanged.
        """A string with no space, '=' or quote renders bare."""
        assert _format_value("train") == "train"

    def test_string_with_space_is_quoted(self):
        # [FUNCTION DOC] Asserts a string with a space is wrapped in double quotes.
        """A string containing a space is double-quoted so it stays one token."""
        assert _format_value("survivors normal") == '"survivors normal"'

    def test_string_with_equals_is_quoted(self):
        # [FUNCTION DOC] Asserts a string containing an equals sign is wrapped in double quotes.
        """A string containing '=' is quoted so it can't be mis-read as a new key."""
        assert _format_value("a=b") == '"a=b"'

    def test_string_with_quote_is_escaped(self):
        # [FUNCTION DOC] Asserts a string with inner double quotes is quoted and each inner quote
        # is preceded by a backslash.
        """A string containing a double-quote is quoted with the inner quote escaped."""
        assert _format_value('say "hi"') == '"say \\"hi\\""'

    def test_string_with_backslash_and_space_escapes_backslash(self):
        # [FUNCTION DOC] Asserts a string with a backslash and a space is quoted and the backslash
        # is doubled.
        """When quoting is triggered, backslashes are escaped too."""
        assert _format_value("a\\b c") == '"a\\\\b c"'

    def test_string_with_newline_is_quoted_and_escaped(self):
        # [FUNCTION DOC] Asserts a real newline becomes the two characters backslash and n inside
        # quotes.
        """A literal newline is escaped so the entry stays on one log line."""
        assert _format_value("a\nb") == '"a\\nb"'

    def test_string_with_carriage_return_is_quoted_and_escaped(self):
        # [FUNCTION DOC] Asserts a carriage return becomes backslash and r inside quotes.
        """A carriage return is escaped so it can't corrupt the log stream."""
        assert _format_value("a\rb") == '"a\\rb"'

    def test_string_with_tab_is_quoted_and_escaped(self):
        # [FUNCTION DOC] Asserts a tab becomes backslash and t inside quotes.
        """A tab is escaped so the value stays one unambiguous token."""
        assert _format_value("a\tb") == '"a\\tb"'

    def test_string_with_only_backslash_is_quoted_and_escaped(self):
        # [FUNCTION DOC] Asserts a string whose only special character is a backslash is still
        # quoted, with the backslash doubled.
        """A backslash alone (no space) still triggers quoting so it round-trips."""
        assert _format_value("a\\b") == '"a\\\\b"'

    def test_string_mixing_all_control_chars(self):
        # [FUNCTION DOC] Asserts a string mixing newline, quotes, tab, backslash and carriage
        # return has every one of them escaped in a single quoted token.
        """Newline, quote, tab and backslash all escape together."""
        assert _format_value('a\n"b"\t\\c\r') == '"a\\n\\"b\\"\\t\\\\c\\r"'

    def test_backslash_escaped_before_other_escapes(self):
        # [FUNCTION DOC] Asserts that a string already containing a backslash followed by n, plus a
        # real newline, yields a doubled backslash then n, then an escaped newline; i.e. backslashes
        # are escaped first so later escapes are not double processed.
        """A pre-escaped-looking string round-trips: the backslash doubles, the newline escapes."""
        assert _format_value("\\n\n") == '"\\\\n\\n"'

    def test_list_element_with_newline_is_quoted(self):
        # [FUNCTION DOC] Asserts a list whose first element holds a newline renders as one quoted,
        # escaped, comma-joined token.
        """A newline inside a list element quotes and escapes the joined value."""
        assert _format_value(["a\nb", "c"]) == '"a\\nb,c"'

    def test_list_of_ints_is_comma_joined_no_space(self):
        # [FUNCTION DOC] Asserts the list 0, 1, 2 renders as comma-joined text without spaces.
        """A list renders comma-joined with no spaces."""
        assert _format_value([0, 1, 2]) == "0,1,2"

    def test_list_of_bools_lowercased(self):
        # [FUNCTION DOC] Asserts a list of a true and a false renders as lowercase true,false.
        """List elements that are bools also render lowercase."""
        assert _format_value([True, False]) == "true,false"

    def test_empty_list_renders_as_empty(self):
        # [FUNCTION DOC] Asserts an empty list renders as the empty string.
        """An empty list renders as an empty value."""
        assert _format_value([]) == ""

    def test_tuple_behaves_like_list(self):
        # [FUNCTION DOC] Asserts a two-element tuple renders identically to the equivalent list.
        """A tuple renders the same as the equivalent list."""
        assert _format_value((0, 1)) == "0,1"

    def test_list_whose_joined_form_has_a_space_is_quoted(self):
        # [FUNCTION DOC] Asserts a list containing an element with a space renders as one quoted
        # comma-joined token.
        """If a list element introduces a space, the whole joined value is quoted."""
        assert _format_value(["a b", "c"]) == '"a b,c"'

    def test_dict_renders_as_quoted_compact_json(self):
        # [FUNCTION DOC] Asserts a one-key mapping renders as compact JSON (no spaces) wrapped in
        # quotes with the inner quotes escaped.
        """A dict value renders as quoted compact JSON."""
        assert _format_value({"a": 1}) == '"{\\"a\\":1}"'


class TestToLogfmt:
    # [CLASS DOC] Tests of the logfmt line builder, which turns a mapping into key=value tokens.
    def test_fields_join_with_space_in_insertion_order(self):
        # [FUNCTION DOC] Asserts three fields render as space separated key=value tokens in the
        # order they were inserted.
        """Fields render as space-separated key=value in insertion order."""
        assert _to_logfmt({"cell": 1, "fn": "train", "ok": True}) == "cell=1 fn=train ok=true"

    def test_empty_fields_render_as_empty_string(self):
        # [FUNCTION DOC] Asserts an empty mapping renders as the empty string.
        """No fields renders as an empty string."""
        assert _to_logfmt({}) == ""

    def test_list_and_empty_fields_keep_one_token_each(self):
        # [FUNCTION DOC] Asserts a list field stays a single comma-joined token and an empty list
        # field renders as a key followed by an equals sign and nothing else.
        """A list field stays one token; an empty field renders as a trailing 'key='."""
        assert _to_logfmt({"alive": [0, 1], "pending": []}) == "alive=0,1 pending="


class TestPruneForLog:
    # [CLASS DOC] Tests of the pruner, which replaces oversized values (measured by compact JSON
    # length against a cap) with short summaries, recursing into mappings.
    def test_small_payload_kept_verbatim(self):
        # [FUNCTION DOC] Asserts a small mapping under a cap of 160 is returned equal to the input.
        """A payload under the cap is returned unchanged."""
        payload = {"quorum_id": 1, "healed": [0]}
        assert prune_for_log(payload, cap=160) == payload

    def test_large_list_field_summarized_small_siblings_kept(self):
        # [FUNCTION DOC] With a cap of 80, asserts a 500-element list field becomes a summary
        # giving the list length while a small sibling field stays as is.
        """An oversized list field becomes a length summary; small siblings stay inline."""
        pruned = prune_for_log({"quorum_id": 1, "checksums": list(range(500))}, cap=80)
        assert pruned == {"quorum_id": 1, "checksums": "<list len=500>"}

    def test_large_string_field_summarized(self):
        # [FUNCTION DOC] With a cap of 80, asserts a 1000-character string field becomes a summary
        # giving the character count.
        """An oversized string field becomes a char-count summary."""
        assert prune_for_log({"blob": "x" * 1000}, cap=80) == {"blob": "<str 1000 chars>"}

    def test_nested_dict_prunes_only_the_big_subfield(self):
        # [FUNCTION DOC] With a cap of 60, asserts that in a nested mapping only the oversized
        # inner list is summarized while the small inner and outer fields stay intact.
        """Recursion summarizes only the oversized nested field, keeping small ones."""
        pruned = prune_for_log({"a": 1, "b": {"big": list(range(500)), "small": 2}}, cap=60)
        assert pruned == {"a": 1, "b": {"big": "<list len=500>", "small": 2}}

    def test_large_tuple_summarized_as_list(self):
        # [FUNCTION DOC] Asserts an oversized 500-element tuple gets the same list-length summary
        # as a list.
        """An oversized tuple is summarized like a list."""
        assert prune_for_log(tuple(range(500)), cap=80) == "<list len=500>"

    def test_value_at_cap_kept_just_over_summarized(self):
        # [FUNCTION DOC] Boundary check with a cap of 10: an 8-character string (10 characters once
        # JSON quoted) is kept, a 9-character one (11 characters as JSON) is summarized with its
        # character count. The comparison is inclusive of the cap.
        """A value whose compact-JSON length is <= cap is kept; one char over is summarized."""
        assert prune_for_log("x" * 8, cap=10) == "x" * 8  # json '"xxxxxxxx"' is exactly 10 chars
        assert prune_for_log("x" * 9, cap=10) == "<str 9 chars>"  # json is 11 chars


class TestLogStructured:
    # [CLASS DOC] Tests of the structured emitter, which formats keyword fields as logfmt (with an
    # optional tag prefix) and logs it through a caller-supplied bound logging method. Each test
    # uses its own named logger and captures records with the log capture fixture.
    def test_emits_tag_prefixed_logfmt_via_given_method(self, caplog):
        # [FUNCTION DOC] Logs through the info method with a tag and four fields and asserts the
        # single message is the tag, a space, then the fields in order as key=value tokens.
        """log_structured prefixes the logfmt line with the caller-supplied tag."""
        logger = logging.getLogger("t_emit")
        with caplog.at_level(logging.INFO, logger="t_emit"):
            log_structured(logger.info, tag="ft", op="execute", phase="start", cell=1, fn="train")
        assert caplog.messages == ["ft op=execute phase=start cell=1 fn=train"]

    def test_no_tag_emits_bare_logfmt(self, caplog):
        # [FUNCTION DOC] Logs without a tag and asserts the message is only the logfmt fields.
        """Without a tag the line is pure logfmt with no prefix."""
        logger = logging.getLogger("t_no_tag")
        with caplog.at_level(logging.INFO, logger="t_no_tag"):
            log_structured(logger.info, op="execute", phase="start")
        assert caplog.messages == ["op=execute phase=start"]

    def test_uses_the_level_of_the_passed_method(self, caplog):
        # [FUNCTION DOC] Logs through the warning method and asserts the captured record has
        # warning level.
        """The record's level is that of the bound method passed (warning here)."""
        logger = logging.getLogger("t_level")
        with caplog.at_level(logging.DEBUG, logger="t_level"):
            log_structured(logger.warning, tag="ft", op="x")
        assert caplog.records[0].levelno == logging.WARNING

    def test_exc_info_is_forwarded(self, caplog):
        # [FUNCTION DOC] Raises and catches a value error, then logs through the error method with
        # the exception-info flag set; asserts the record carries that exception type.
        """exc_info=True attaches the active exception to the record."""
        logger = logging.getLogger("t_exc")
        with caplog.at_level(logging.ERROR, logger="t_exc"):
            try:
                raise ValueError("boom")
            except ValueError:
                log_structured(logger.error, tag="ft", op="x", exc_info=True)
        assert caplog.records[0].exc_info[0] is ValueError

    def test_stacklevel_points_to_caller_not_helper(self, caplog):
        # [FUNCTION DOC] Asserts the captured record's source file name is this test file, i.e. the
        # emitter adjusts the stack level so the caller is reported, not the helper module.
        """stacklevel=2 makes the record's filename the caller's, not structured_log.py."""
        logger = logging.getLogger("t_stack")
        with caplog.at_level(logging.INFO, logger="t_stack"):
            log_structured(logger.info, tag="ft", op="x")
        assert caplog.records[0].filename == "test_structured_log.py"


class FakeActor:
    # [CLASS DOC] HELPER: Test double whose four methods are wrapped by the logging decorator: a
    # sync method, a sync method that raises, an async method, and an async method that never
    # completes unless cancelled.
    @with_logs
    def train(self):
        # [FUNCTION DOC] HELPER: Sync method; returns the integer 42.
        return 42

    @with_logs
    def update_weights(self):
        # [FUNCTION DOC] HELPER: Sync method; always raises a value error with message boom.
        raise ValueError("boom")

    @with_logs
    async def send_ckpt(self):
        # [FUNCTION DOC] HELPER: Async method; returns the string done.
        return "done"

    @with_logs
    async def wait_forever(self):
        # [FUNCTION DOC] HELPER: Async method; waits on an event that is never set, so it only
        # ends through cancellation.
        await asyncio.Event().wait()


class TestWithLogs:
    # [CLASS DOC] Tests of the logging decorator: it logs a start line and an end line (with
    # success flag and elapsed seconds), reports the runtime class and method name, logs errors
    # with traceback and re-raises, handles async methods and cancellation, and keeps metadata.
    def test_sync_method_emits_start_then_end_with_class_and_method(self, caplog):
        # [FUNCTION DOC] Calls the sync method and asserts it returns 42, the first message is the
        # actor-tagged start line with class and function names, and the second is the end line
        # with success true followed by elapsed seconds.
        """A sync call emits phase=start then phase=end carrying the auto-detected class and method."""
        with caplog.at_level(logging.INFO):
            assert FakeActor().train() == 42
        assert caplog.messages[0] == "actor cls=FakeActor fn=train phase=start"
        assert caplog.messages[1].startswith("actor cls=FakeActor fn=train phase=end ok=true elapsed_s=")

    def test_class_is_read_from_the_runtime_instance(self, caplog):
        # [FUNCTION DOC] Defines a subclass of the fake actor, calls the inherited train method,
        # and asserts the start line names the subclass, not the defining class.
        """The cls field reflects the actual runtime instance, not where the method was defined."""

        class SubActor(FakeActor):
            pass

        with caplog.at_level(logging.INFO):
            SubActor().train()
        assert caplog.messages[0] == "actor cls=SubActor fn=train phase=start"

    def test_exception_logs_end_not_ok_with_exc_info_and_reraises(self, caplog):
        # [FUNCTION DOC] Calls the raising method, asserts the value error propagates, the first
        # message is the start line, and the second record is an error-level end line with success
        # false, elapsed seconds, and the value error exception info attached.
        """On exception the end line is ok=false with the traceback, and the error propagates."""
        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError, match="boom"):
                FakeActor().update_weights()
        assert caplog.messages[0] == "actor cls=FakeActor fn=update_weights phase=start"
        end = caplog.records[1]
        assert end.getMessage().startswith("actor cls=FakeActor fn=update_weights phase=end ok=false elapsed_s=")
        assert end.levelno == logging.ERROR
        assert end.exc_info[0] is ValueError

    def test_async_method_emits_start_then_end_ok(self, caplog):
        # [FUNCTION DOC] Runs the async method to completion in a fresh event loop; asserts the
        # result is done, then a start line and a successful end line with elapsed seconds.
        """An async call emits the same start/end pair after the coroutine completes."""
        with caplog.at_level(logging.INFO):
            assert asyncio.run(FakeActor().send_ckpt()) == "done"
        assert caplog.messages[0] == "actor cls=FakeActor fn=send_ckpt phase=start"
        assert caplog.messages[1].startswith("actor cls=FakeActor fn=send_ckpt phase=end ok=true elapsed_s=")

    def test_async_cancellation_logs_cancelled_at_info_and_reraises(self, caplog):
        # [FUNCTION DOC] Starts the never-ending async method as a task, yields once so it starts,
        # cancels it, and asserts awaiting it raises the cancellation error (inside a nested
        # coroutine run in a fresh event loop). Then asserts the second record is an info-level end
        # line with success false, elapsed seconds, a trailing cancelled=true marker, and no
        # exception info.
        """Cancelling an async call emits an info end line with cancelled=true and no traceback."""

        async def run() -> None:
            task = asyncio.ensure_future(FakeActor().wait_forever())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with caplog.at_level(logging.INFO):
            asyncio.run(run())
        end = caplog.records[1]
        assert end.getMessage().startswith("actor cls=FakeActor fn=wait_forever phase=end ok=false elapsed_s=")
        assert end.getMessage().endswith("cancelled=true")
        assert end.levelno == logging.INFO
        assert not end.exc_info

    def test_preserves_wrapped_function_metadata(self):
        # [FUNCTION DOC] Asserts the decorated method still reports its original name.
        """functools.wraps keeps the original __name__ so Ray/introspection still sees it."""
        assert FakeActor.train.__name__ == "train"
