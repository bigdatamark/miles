# [MODULE DOC]
# CPU tests that the two engine-update code paths (distributed broadcast and colocated CUDA IPC)
# send the engine request with the cache flush decision left off explicitly, using a fake API client
# that records every call as method name and keyword arguments. The distributed module is patched
# with a mock so no collective runs.
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.utils import async_utils

_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"
_TENSOR_MODULE = "miles.backends.training_utils.weight_update.protocols.cuda_ipc"

_NAMED_TENSORS = [("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, 2))]


class _RecordingApiClient:
    # [CLASS DOC] HELPER: Fake engine API client accepting any method name; each call is logged as
    # name and keyword mapping and returns a success result.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the empty call log.
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name: str):
        # [FUNCTION DOC] HELPER: Returns an async function standing in for the named engine method.
        async def method(**kwargs):
            # [FUNCTION DOC] HELPER: Logs the call and returns a mapping with success true.
            self.calls.append((name, kwargs))
            return {"success": True}

        return method


def test_distributed_update_does_not_flush():
    # [FUNCTION DOC] Calls the distributed update function with group name "miles-pp_0", a mock
    # group, one recording client and a one-tensor list (collective broadcast mocked), then waits
    # for the returned futures. Asserts the first recorded call is the engine's distributed update
    # carrying the group name, and that its flush cache argument is absent or false.
    """The trainer, not a default, decides that a broadcast update skips the cache flush."""
    from miles.backends.training_utils.weight_update.protocols.broadcast import update_weights_from_distributed

    client = _RecordingApiClient()

    with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
        dist_mock.broadcast.return_value = MagicMock()
        futures = update_weights_from_distributed(
            group_name="miles-pp_0",
            group=MagicMock(),
            rollout_engines=[client],
            converted_named_tensors=_NAMED_TENSORS,
        )
        async_utils.wait_futures(futures)

    name, kwargs = client.calls[0]
    assert name == "update_weights_from_distributed"
    assert kwargs["group_name"] == "miles-pp_0"
    assert kwargs.get("flush_cache", False) is False


def test_colocated_base_update_does_not_flush():
    # [FUNCTION DOC] Skipped if SGLang is not importable. Sends the one-tensor list through the
    # colocated send function with a recording client as the IPC engine, a single-process world
    # (rank zero, size one, a fake object gather that copies the local object into slot zero) and a
    # mock gather group, then waits for the futures. Asserts the first recorded call is the engine's
    # update from tensor with the flattened bucket load format and an absent or false flush cache.
    """Same explicit no-flush decision on the colocated IPC path."""
    pytest.importorskip("sglang")

    from miles.backends.training_utils.weight_update.protocols.cuda_ipc import _send_to_colocated_engine

    client = _RecordingApiClient()

    def fake_gather_object(obj, object_gather_list=None, dst=None, group=None):
        # [FUNCTION DOC] HELPER: Fake single-process gather; stores the local object in slot zero of
        # the output list, ignoring destination and group.
        object_gather_list[0] = obj

    with patch(f"{_TENSOR_MODULE}.dist") as dist_mock:
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        dist_mock.gather_object.side_effect = fake_gather_object
        futures, _long_live_tensors = _send_to_colocated_engine(
            _NAMED_TENSORS,
            ipc_engine=client,
            ipc_gather_src=0,
            ipc_gather_group=MagicMock(),
        )
        async_utils.wait_futures(futures)

    name, kwargs = client.calls[0]
    assert name == "update_weights_from_tensor"
    assert kwargs["load_format"] == "flattened_bucket"
    assert kwargs.get("flush_cache", False) is False
