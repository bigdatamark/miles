# [MODULE DOC] CPU tests for registering replay data on a list of replay objects: each replay
# records one slice of the replay tensor (shape tokens by streams by experts-per-token) taken along
# the stream axis, chosen by its stream index when set, else by its position in the list; an
# out-of-range stream index raises an assertion error. Registered as a CPU test in the first CPU
# stage.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.training_utils.replay.data import register_replay_list_sequential


class _Replay:
    # [CLASS DOC] HELPER: Fake replay holding an optional stream index and a list of recorded values.
    def __init__(self, stream_idx=None):
        # [FUNCTION DOC] HELPER: Stores the stream index and starts an empty recorded list.
        self.stream_idx = stream_idx
        self.recorded = []

    def record(self, value):
        # [FUNCTION DOC] HELPER: Appends the value to the recorded list.
        self.recorded.append(value)


def test_register_replay_list_sequential_falls_back_to_enum_when_no_stream_idx():
    # [FUNCTION DOC] Test: three replays without stream index and data of shape 5 by 3 by 2: each
    # records exactly one value, equal to the slice at its list position along the second axis.
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    replays = [_Replay(), _Replay(), _Replay()]

    register_replay_list_sequential(replays, replay_data)

    for replay_idx, replay in enumerate(replays):
        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay_idx])


def test_register_replay_list_sequential_uses_stream_idx_when_set():
    # [FUNCTION DOC] Test: two replays with stream indices 1 and 3 and data of shape 5 by 4 by 2:
    # each records the slice at its own stream index along the second axis.
    # PP case: 2 local modules at global streams 1 and 3 out of 4.
    replay_data = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    replays = [_Replay(stream_idx=1), _Replay(stream_idx=3)]

    register_replay_list_sequential(replays, replay_data)

    torch.testing.assert_close(replays[0].recorded[0], replay_data[:, 1])
    torch.testing.assert_close(replays[1].recorded[0], replay_data[:, 3])


def test_register_replay_list_sequential_rejects_out_of_range_stream_idx():
    # [FUNCTION DOC] Test: a replay with stream index 4 against data with only 4 streams raises an
    # assertion error mentioning "out of range".
    replay_data = torch.zeros(5, 4, 2)
    replays = [_Replay(stream_idx=4)]

    with pytest.raises(AssertionError, match="out of range"):
        register_replay_list_sequential(replays, replay_data)
