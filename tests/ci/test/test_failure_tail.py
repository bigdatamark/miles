# [MODULE DOC]
# Tests for the failure-tail feature of the CI suite runner (tests.ci.ci_utils): while streaming a
# child test process's output the runner passes every byte through to stdout but keeps only a
# bounded tail of recent lines, and on failure prints that tail in the final summary so the
# job log's last bytes explain the failure even when the log is tens of megabytes. Registers itself
# as a CPU test in suite stage-a-cpu. Defines NOTIFIER_TAIL_BYTES (33600), the size of the window the
# failure analysis downloads from the end of a job log.
"""The end of a job log must explain the failure, whatever the log's size.

A GPU suite writes tens of megabytes after the test that failed, so the
notifier's tail window never reaches the traceback. The suite runner already
sees every line, so it keeps the failing test's last output and repeats it in
the summary it prints last.
"""

import io
import logging
import sys
from collections import deque
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci
from tests.ci.ci_utils import FAILURE_TAIL_BYTES, FAILURE_TAIL_LINES
from tests.ci.ci_utils import TestFile as CITestFile
from tests.ci.ci_utils import _drain_child_output, _failure_tail, run_unittest_files

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# The window ci_failure_analysis downloads from the end of a job log.
NOTIFIER_TAIL_BYTES = 33_600


class _CapturedStdout:
    # [CLASS DOC] HELPER: Minimal stand-in for standard output exposing only a binary in-memory
    # buffer attribute, which is what the drain function writes to.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Create an empty in-memory byte buffer as the buffer attribute.
        self.buffer = io.BytesIO()


def drain(body: bytes) -> tuple[bytes, deque]:
    # [FUNCTION DOC] HELPER: Feed the given bytes through the runner's child-output drain function
    # with an empty bounded line deque (capacity eight times the failure tail line limit), while
    # temporarily replacing standard output by the capturing stand-in (restored in a finally block,
    # even on error). Returns a pair: all bytes written to the captured stdout, and the deque.
    tail: deque = deque(maxlen=FAILURE_TAIL_LINES * 8)
    captured = _CapturedStdout()
    real, sys.stdout = sys.stdout, captured
    try:
        _drain_child_output(io.BytesIO(body), tail)
    finally:
        sys.stdout = real
    return captured.buffer.getvalue(), tail


def test_draining_passes_every_byte_through_untouched():
    # [FUNCTION DOC] Drains 50000 numbered lines and asserts the captured stdout equals the input
    # byte for byte.
    body = b"".join(f"line {index}\n".encode() for index in range(50_000))
    passed_through, _ = drain(body)
    assert passed_through == body


def test_draining_holds_a_bounded_tail_of_a_large_stream():
    # [FUNCTION DOC] Drains 40 million bytes of a single repeated character (one huge line without
    # newline) and asserts the retained tail holds fewer than one million bytes in total.
    body = b"x" * 40_000_000
    _, tail = drain(body)
    assert sum(len(chunk) for chunk in tail) < 1_000_000


def test_the_tail_keeps_the_last_lines_and_drops_blank_ones():
    # [FUNCTION DOC] Renders a deque containing a line, a blank line, a whitespace-only line, an
    # assertion line and an exit line; asserts the rendered lines are the three non-blank ones in
    # order. Also renders a deque with one chunk of 500 short lines and asserts at most the failure
    # tail line limit of lines are kept.
    tail = deque([b"early\n", b"\n   \n", b"AssertionError: boom\n", b"exit 1\n"])
    assert _failure_tail(tail).splitlines() == ["early", "AssertionError: boom", "exit 1"]
    assert len(_failure_tail(deque([b"l\n" * 500])).splitlines()) <= FAILURE_TAIL_LINES


def test_a_failing_test_explains_itself_inside_the_notifier_window(tmp_path, monkeypatch, caplog):
    # [FUNCTION DOC] End-to-end check of the runner. Writes a script that prints 5000 noise lines then
    # raises an assertion error with a distinctive message, changes into the temporary directory and
    # runs the suite runner on it with a 120 second timeout and continue-on-error, capturing
    # info-level logs. Asserts the log says the file returned exit code 1, that a summary section
    # follows the last "FAILED:" header line, that the summary contains the assertion message, and
    # that the summary is shorter than the notifier window. Spawns a subprocess.
    Path(tmp_path / "test_boom.py").write_text(
        "for index in range(5000):\n"
        "    print(f'noise {index}')\n"
        "raise AssertionError('the reactor exploded at step 42')\n"
    )
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        run_unittest_files([CITestFile(name="test_boom.py")], timeout_per_file=120, continue_on_error=True)
    assert "FAILED: test_boom.py returned exit code 1" in caplog.text
    _, separator, summary = caplog.text.rpartition("\nFAILED:\n")
    assert separator, "the runner printed no failure summary"
    assert "the reactor exploded at step 42" in summary
    assert len(summary) < NOTIFIER_TAIL_BYTES


def test_the_summary_survives_a_log_far_larger_than_the_notifier_window():
    # [FUNCTION DOC] Builds a stream of 400000 progress lines followed by a final assertion line
    # (over a hundred times the notifier window), drains it, renders the tail, and asserts the
    # rendered text contains the final assertion line, its encoded size does not exceed the failure
    # tail byte limit, and the input was indeed more than 100 times the notifier window.
    body = b"".join(f"progress step {index}\n".encode() for index in range(400_000))
    body += b"AssertionError: the reactor exploded\n"
    _, tail = drain(body)
    rendered = _failure_tail(tail)
    assert "AssertionError: the reactor exploded" in rendered
    assert len(rendered.encode()) <= FAILURE_TAIL_BYTES
    assert len(body) > 100 * NOTIFIER_TAIL_BYTES
