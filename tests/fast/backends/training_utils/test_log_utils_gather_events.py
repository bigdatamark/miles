# [MODULE DOC] CPU tests of the fault-tolerance log events emitted around the cross-cell log data
# gather in the training metrics log utilities: a successful gather emits a start record and an
# end record marked success, both tagged with the cross-cell operation, kind and rank; a runtime
# error from the collective is swallowed (returns none) and the end record reports failure with a
# degraded flag at warning level carrying the exception info. Uses a fake multi-process-group
# utility and a fake non-source rank (rank 1 of 4).
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.training_utils.metrics import log_utils

_LOG_UTILS_LOGGER = "miles.backends.training_utils.metrics.log_utils"


class _FakeMultiPGUtil:
    # [CLASS DOC] HELPER: Fake of the multi-process-group gather utility; records the objects passed
    # to gather and either raises a configured error or returns the configured gathered value.
    def __init__(self, gathered: list[dict] | None, error: Exception | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the value to return, the optional error to raise, and an
        # empty list of gathered inputs.
        self.gathered = gathered
        self.error = error
        self.calls: list[dict] = []

    def gather_object(self, obj, groups_inner_to_outer):
        # [FUNCTION DOC] HELPER: Records the object; raises the configured error if any, else
        # returns the configured gathered value. The groups argument is ignored.
        self.calls.append(obj)
        if self.error is not None:
            raise self.error
        return self.gathered


@pytest.fixture()
def non_source_rank(monkeypatch) -> None:
    # [FUNCTION DOC] HELPER: Fixture; patches the log utilities' parallel state getter to a fake
    # whose effective data-parallel-with-cp axis has rank 1, size 4 and no gloo groups, i.e. a rank
    # that is not the source of the gather.
    parallel_state = SimpleNamespace(
        effective_dp_cp=SimpleNamespace(rank=1, size=4, gloo_groups_inner_to_outer=[]),
    )
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)


def _gather_messages(caplog) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the formatted messages of captured log records emitted by the
    # log utilities' logger, in order.
    return [record.getMessage() for record in caplog.records if record.name == _LOG_UTILS_LOGGER]


class TestGatherLogDataEvents:
    # [CLASS DOC] Tests for the log events of the log data gather function.
    def test_successful_gather_brackets_the_collective_with_ft_records(self, non_source_rank, monkeypatch, caplog):
        # [FUNCTION DOC] Test: with the fake gather returning none, gathering ("perf" metric, empty
        # args, step 3, one-entry dict) at info level returns none, and the exact two messages are a
        # start record and an end record with success true, tagged operation cross_cell, kind
        # log_gather, rank 1.
        """A successful gather emits ft-tagged start and success end log_gather records for the rank."""
        monkeypatch.setattr(log_utils, "MultiPGUtil", _FakeMultiPGUtil(gathered=None))

        with caplog.at_level(logging.INFO, logger=_LOG_UTILS_LOGGER):
            result = log_utils.gather_log_data("perf", Namespace(), 3, {"loss": 1.0})

        assert result is None
        assert _gather_messages(caplog) == [
            "ft op=cross_cell phase=start kind=log_gather rank=1",
            "ft op=cross_cell phase=end kind=log_gather rank=1 success=true",
        ]

    def test_gather_runtime_error_is_reported_as_a_degraded_ft_record(self, non_source_rank, monkeypatch, caplog):
        # [FUNCTION DOC] Test: with the fake gather raising a runtime error, gathering returns none
        # (error swallowed); messages are the start record then an end record with success false and
        # degraded true; the last record from the utilities' logger is at warning level and its
        # exception info has the runtime error type.
        """A RuntimeError from the collective ends the ft record as a degraded failure with the traceback."""
        monkeypatch.setattr(log_utils, "MultiPGUtil", _FakeMultiPGUtil(gathered=None, error=RuntimeError("boom")))

        with caplog.at_level(logging.INFO, logger=_LOG_UTILS_LOGGER):
            result = log_utils.gather_log_data("perf", Namespace(), 3, {"loss": 1.0})

        assert result is None
        assert _gather_messages(caplog) == [
            "ft op=cross_cell phase=start kind=log_gather rank=1",
            "ft op=cross_cell phase=end kind=log_gather rank=1 success=false degraded=true",
        ]
        end = [record for record in caplog.records if record.name == _LOG_UTILS_LOGGER][-1]
        assert end.levelno == logging.WARNING
        assert end.exc_info[0] is RuntimeError
