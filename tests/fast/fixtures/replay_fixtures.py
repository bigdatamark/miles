# [MODULE DOC]
# Shared helpers for routing-replay tests that run on cpu: a device-neutral replay stand-in, a helper
# that installs an architecture hook and registers a fresh replay as current on the global routing
# replay manager, and a helper that resets that manager to a known state.
"""Shared helpers for routing-replay tests."""

from miles.utils.replay_base import routing_replay_manager


class CpuReplay:
    # [CLASS DOC] HELPER: Cpu replay recorder. Keeps a list of recorded top-index tensors and two
    # independent read cursors, one for forward pops and one for backward pops; neither pops remove
    # entries. Forward cursor can be reset; the backward one cannot.
    """Device-neutral stand-in for ``Replay``.

    The real one pins its host buffer and returns indices on ``torch.cuda.current_device()``;
    routing-replay suites are registered stage-a-cpu. The two independent cursors are the
    behaviour under test and are reproduced exactly.
    """

    def __init__(self, stream_idx=0):
        # [FUNCTION DOC] HELPER: Stores the stream index (default 0), an empty recorded list and both
        # cursors at zero.
        self.stream_idx = stream_idx
        self.recorded = []
        self.forward_index = 0
        self.backward_index = 0

    def record(self, top_indices):
        # [FUNCTION DOC] HELPER: Appends a detached clone of the tensor to the recorded list.
        self.recorded.append(top_indices.detach().clone())

    def pop_forward(self):
        # [FUNCTION DOC] HELPER: Returns the recorded tensor at the forward cursor and advances it by
        # one; raises an index error when past the end.
        top_indices = self.recorded[self.forward_index]
        self.forward_index += 1
        return top_indices

    def pop_backward(self):
        # [FUNCTION DOC] HELPER: Same as the forward pop but using the independent backward cursor.
        top_indices = self.recorded[self.backward_index]
        self.backward_index += 1
        return top_indices

    def clear_forward(self):
        # [FUNCTION DOC] HELPER: Resets only the forward cursor to zero; recordings are kept.
        self.forward_index = 0


def wire_replay(module, install) -> CpuReplay:
    # [FUNCTION DOC] HELPER: Calls the install callback on the module (installing the architecture
    # hook), creates a fresh cpu replay, appends it to the global manager's replay list, makes it the
    # manager's current replay, and returns it. Mutates the global manager.
    """Install an arch hook on ``module`` and make a fresh CPU replay the current stream."""
    install(module)
    replay = CpuReplay()
    routing_replay_manager.replays.append(replay)
    routing_replay_manager.set_current(replay)
    return replay


def reset_routing_replay_manager(enabled: bool) -> None:
    # [FUNCTION DOC] HELPER: Resets the global manager: sets enabled to the argument, turns the replay
    # result check off, empties the replay list, clears the current replay, and sets the stage to
    # fallthrough.
    routing_replay_manager.enabled = enabled
    routing_replay_manager.enable_check_replay_result = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    routing_replay_manager.stage = "fallthrough"
