# [MODULE DOC] Tests for the argv manipulator: reading every value of a flag (space or equals
# sign spelling, in order, ignoring flags embedded in other flags' values, refusing a trailing
# flag with no value), the effective (last) value, whether a flag is defined at all (any
# spelling, but not a longer flag sharing its prefix), and setting a flag (append when absent,
# replace the last declaration in place keeping its spelling, never mutating the input list).
import pytest

from miles.utils.external_utils.command_utils.common import ArgvManipulator


class TestGet:
    # [CLASS DOC] Tests of the get method, which returns a list of all values given for a flag.
    def test_reads_the_space_form(self):
        # [FUNCTION DOC] A flag followed by a separate value token yields a list with that value.
        """This is how a launch script writes a flag, and the value is the next token."""
        assert ArgvManipulator.get(["--train-backend", "fsdp"], "--train-backend") == ["fsdp"]

    def test_reads_the_equals_form(self):
        # [FUNCTION DOC] A flag joined to its value by an equals sign yields a list with the value.
        """`--flag=value` is one token, and argparse takes it, so a reader that misses it reads a stale argv."""
        assert ArgvManipulator.get(["--train-backend=fsdp"], "--train-backend") == ["fsdp"]

    def test_reads_every_occurrence_in_order(self):
        # [FUNCTION DOC] Two occurrences (separate, then equals sign) yield both values in order.
        """argparse takes the last of a repeated flag, so a caller has to be able to see that there were two."""
        argv = ["--train-backend", "megatron", "--train-backend=fsdp"]

        assert ArgvManipulator.get(argv, "--train-backend") == ["megatron", "fsdp"]

    def test_does_not_mistake_another_flags_value_for_the_flag(self):
        # [FUNCTION DOC] A token for another flag whose value contains the flag text yields an
        # empty list.
        """A substring search finds the flag inside `--data-path=--train-backend fsdp` and reads a value nobody set."""
        assert ArgvManipulator.get(["--data-path=--train-backend fsdp"], "--train-backend") == []

    def test_refuses_a_trailing_flag_that_names_nothing(self):
        # [FUNCTION DOC] An argv ending with the flag as its last token raises an assertion error
        # mentioning the last argument.
        """argparse would fail on this argv inside the pod, where the failure is much harder to read."""
        with pytest.raises(AssertionError, match="last argument"):
            ArgvManipulator.get(["--rollout-num-gpus", "8", "--train-backend"], "--train-backend")


class TestGetEffective:
    # [CLASS DOC] Tests of the effective value method (the last declared value, or none).
    def test_takes_the_last_declaration(self):
        # [FUNCTION DOC] With an equals spelling followed by a separate spelling, the later value
        # is returned.
        """argparse keeps the last one, so a caller deriving a backend from any earlier one reads a value nobody ran."""
        argv = ["--train-backend=fsdp", "--train-backend", "megatron"]

        assert ArgvManipulator.get_effective(argv, "--train-backend") == "megatron"

    def test_answers_none_for_a_flag_the_argv_never_named(self):
        # [FUNCTION DOC] An argv without the flag gives none.
        """The caller has to be able to tell an unset flag from one set to the empty string."""
        assert ArgvManipulator.get_effective(["--rollout-num-gpus", "8"], "--train-backend") is None


class TestIsDefined:
    # [CLASS DOC] Tests of the defined check, which says whether an argv mentions a flag.
    @pytest.mark.parametrize("argv", [["--run-uuid", "abc"], ["--run-uuid=abc"], ["--run-uuid"]])
    def test_sees_a_flag_however_it_is_written(self, argv):
        # [FUNCTION DOC] Parametrized over the separate, equals sign and bare trailing spellings;
        # all count as defined (and, unlike the get method, the bare one does not raise).
        """It answers whether the argv already speaks about the flag, not what it says about it."""
        assert ArgvManipulator.is_defined(argv, "--run-uuid")

    def test_does_not_see_a_flag_that_is_only_a_prefix_of_another(self):
        # [FUNCTION DOC] An argv with only a longer flag that starts with the asked flag is not
        # defined.
        """`--run-uuid-file` is a different flag, and treating it as this one would skip setting this one."""
        assert not ArgvManipulator.is_defined(["--run-uuid-file", "/tmp/x"], "--run-uuid")


class TestSet:
    # [CLASS DOC] Tests of the set method, which returns a new argv with a flag value set.
    def test_appends_the_flag_an_argv_never_named(self):
        # [FUNCTION DOC] An argv without the flag gets the flag and value appended at the end.
        """A run that never asked for the flag still has to leave with the value the launcher computed."""
        rewritten = ArgvManipulator.set(["--rollout-num-gpus", "8"], "--run-uuid", "abc")

        assert rewritten == ["--rollout-num-gpus", "8", "--run-uuid", "abc"]

    def test_overwrites_a_value_the_argv_already_carried(self):
        # [FUNCTION DOC] An existing separate-spelling value is replaced in place, keeping the
        # flag's position.
        """The launcher's value wins, and it has to win in place so the flag keeps its position."""
        argv = ["--run-uuid", "old", "--rollout-num-gpus", "8"]

        rewritten = ArgvManipulator.set(argv, "--run-uuid", "new")

        assert rewritten == ["--run-uuid", "new", "--rollout-num-gpus", "8"]

    def test_overwrites_a_flag_the_argv_wrote_as_one_token(self):
        # [FUNCTION DOC] An equals-spelled flag keeps its equals spelling with the new value.
        """`--flag=value` is a legal argv, and appending a second declaration would be read instead of it."""
        rewritten = ArgvManipulator.set(["--run-uuid=old", "--rollout-num-gpus", "8"], "--run-uuid", "new")

        assert rewritten == ["--run-uuid=new", "--rollout-num-gpus", "8"]

    def test_overwrites_the_last_declaration_because_that_is_the_one_argparse_takes(self):
        # [FUNCTION DOC] With two declarations, only the last is rewritten; the first is untouched.
        """Rewriting the first one leaves the pod running on the stale value the launcher meant to replace."""
        argv = ["--run-uuid", "first", "--rollout-num-gpus", "8", "--run-uuid=second"]

        rewritten = ArgvManipulator.set(argv, "--run-uuid", "new")

        assert rewritten == ["--run-uuid", "first", "--rollout-num-gpus", "8", "--run-uuid=new"]

    def test_does_not_mutate_what_it_was_given(self):
        # [FUNCTION DOC] After calling set on a list, the original list is unchanged.
        """The caller keeps its own argv, and a launcher builds several from one list."""
        argv = ["--rollout-num-gpus", "8"]

        ArgvManipulator.set(argv, "--run-uuid", "abc")

        assert argv == ["--rollout-num-gpus", "8"]
