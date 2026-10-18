# [MODULE DOC]
# Tests for the response repetition detector in the metric utilities module. The detector scans a
# response text in fixed-size character windows (10,000 characters) advanced by a stride, plus
# always the first and the final window, and flags the text if any window's compression ratio
# exceeds a strict threshold (a ratio of exactly 10 is not flagged). Texts shorter than one window
# are never scanned. The number of windows is capped by a module constant so very long responses
# cost bounded work, while still covering the first and final windows. Most tests replace the
# compression ratio function with a mock returning a (ratio, extra) pair; one uses real compression.
import hashlib
from unittest.mock import Mock

from miles.utils import metric_utils


def test_has_repetition_checks_stride_aligned_middle_window(monkeypatch) -> None:
    # [FUNCTION DOC] A 20,000-character text (5,000 "a", 10,000 "b", 5,000 "c") with a mocked ratio
    # that returns 11 only for the all-"b" window: the detector flags repetition and calls the
    # ratio function exactly twice: the first window (offset 0) and then the next stride-aligned
    # window, which starts at offset 5,000 and is entirely "b" (so the stride appears to be 5,000
    # characters, and scanning stops at the first flagged window).
    text = "a" * 5_000 + "b" * 10_000 + "c" * 5_000
    compression_ratio = Mock(side_effect=lambda window: (11.0, 0.0) if window == "b" * 10_000 else (1.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert metric_utils.has_repetition(text)
    assert compression_ratio.call_count == 2


def test_has_repetition_checks_unaligned_final_window(monkeypatch) -> None:
    # [FUNCTION DOC] A 12,345-character text whose last 10,000 characters are repetitive, with the
    # mock returning 11 only for exactly that final window: repetition is detected with two
    # ratio calls (the first window and the unaligned final window).
    text = "a" * 2_345 + "b" * 10_000
    final_window = text[-10_000:]
    compression_ratio = Mock(side_effect=lambda window: (11.0, 0.0) if window == final_window else (1.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert metric_utils.has_repetition(text)
    assert compression_ratio.call_count == 2


def test_has_repetition_checks_exactly_ten_thousand_characters() -> None:
    # [FUNCTION DOC] With real compression, a text of exactly 10,000 characters (a repeated
    # six-letter word 1,666 times plus four filler characters) is scanned and flagged.
    assert metric_utils.has_repetition("repeat" * 1_666 + "xxxx")


def test_has_repetition_ignores_short_text(monkeypatch) -> None:
    # [FUNCTION DOC] A 9,999-character text is never flagged and the ratio function is not called,
    # even though the mock would report a huge ratio.
    compression_ratio = Mock(return_value=(100.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert not metric_utils.has_repetition("x" * 9_999)
    compression_ratio.assert_not_called()


def test_has_repetition_uses_strict_threshold(monkeypatch) -> None:
    # [FUNCTION DOC] A ratio of exactly 10.0 for a 10,000-character text is not flagged (the
    # comparison is strictly greater than).
    monkeypatch.setattr(metric_utils, "compression_ratio", lambda _: (10.0, 0.0))

    assert not metric_utils.has_repetition("x" * 10_000)


def test_repetition_window_count_is_bounded_for_very_long_text(monkeypatch) -> None:
    # [FUNCTION DOC] A 2,000,000-character text with a mocked low ratio is not flagged, and the
    # number of ratio calls does not exceed the module's maximum window count constant.
    """The scan runs on the rollout executor's event loop; its cost must not
    scale with response length."""
    compression_ratio = Mock(return_value=(1.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert not metric_utils.has_repetition("x" * 2_000_000)  # ~400 natural stride windows
    assert compression_ratio.call_count <= metric_utils.REPETITION_MAX_WINDOWS


def test_bounded_windows_keep_first_and_final_coverage() -> None:
    # [FUNCTION DOC] Generate the window list for a 500,000-character varied text with the private
    # window generator; assert its length is within the cap, the first window is the text prefix of
    # window size, and the last window is the text suffix of window size.
    text = "".join(chr(33 + (i % 90)) for i in range(500_000))
    windows = list(metric_utils._repetition_windows(text))

    assert len(windows) <= metric_utils.REPETITION_MAX_WINDOWS
    assert windows[0] == text[: metric_utils.REPETITION_WINDOW_SIZE_CHARS]
    assert windows[-1] == text[-metric_utils.REPETITION_WINDOW_SIZE_CHARS :]


def test_repetition_mid_way_in_a_very_long_response_is_still_caught() -> None:
    # [FUNCTION DOC] With real compression: build a roughly 275,000-character low-compressibility
    # filler from hashes, and insert a 25,000-character repeated "spam" run at offset 150,000.
    # The detector flags the text with the run, and does not flag the filler alone.
    """Up to ~320k chars the widened stride keeps windows overlapping, so a
    repetitive run longer than one window always lands fully inside some
    window (real compression, no mocks)."""
    filler = "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(4_300))  # ~275k, low ratio
    text = filler[:150_000] + "spam" * 6_250 + filler[150_000:]

    assert metric_utils.has_repetition(text)
    assert not metric_utils.has_repetition(filler)
