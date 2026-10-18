# [MODULE DOC]
# CPU tests of the exact engine API calls the disk-delta protocol issues when reloading engines and
# when capturing the baseline checkpoint, using a fake recording API client and a protocol object
# built without its constructor. Covers call order and arguments, the in-place pause mode skipping
# the cache flush, non-source ranks issuing no requests, and barrier use on non-source ranks.
from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


class _RecordingApiClient:
    # [CLASS DOC] HELPER: Fake engine API client accepting any method name; every call is appended
    # as a pair of method name and keyword arguments to a shared list and returns a success result.
    def __init__(self, calls: list[tuple[str, dict]]):
        # [FUNCTION DOC] HELPER: Stores the shared call log.
        self._calls = calls

    def __getattr__(self, name: str):
        # [FUNCTION DOC] HELPER: Returns an async function standing in for the named engine method.
        async def method(**kwargs):
            # [FUNCTION DOC] HELPER: Logs the call and returns a mapping with success true.
            self._calls.append((name, kwargs))
            return {"success": True}

        return method


def _make_protocol(calls: list[tuple[str, dict]]) -> UpdateWeightFromDiskDelta:
    # [FUNCTION DOC] HELPER: Builds a disk-delta protocol bypassing its constructor, with args for
    # local checkpoint dir "/local/ckpt", shared dir "/shared/delta", retract pause mode and no
    # equality check, one recording engine logging into the given list, no post-write hook and
    # version dir "/shared/delta/v7". Returns it.
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace(
        update_weight_local_checkpoint_dir="/local/ckpt",
        update_weight_disk_dir="/shared/delta",
        pause_generation_mode="retract",
        check_weight_update_equal=False,
    )
    protocol.rollout_engines = [_RecordingApiClient(calls)]
    protocol._post_write_hook = None
    protocol._version_dir = "/shared/delta/v7"
    return protocol


def test_reload_engines_pulls_with_both_checkpoint_dirs_then_reloads():
    # [FUNCTION DOC] Reloads engines for version seven as rank zero (distributed module and gloo
    # group patched). Asserts the call order: pull weights, pause generation, flush cache, update
    # from disk, continue generation; the pause call carries the retract mode; the pull carries the
    # target version seven, the local checkpoint dir and the shared source dir; the disk update
    # carries the local checkpoint dir as model path and the weight version as the string "7".
    """The reload pull carries both checkpoint dirs the deleted engine wrapper used to inject."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        protocol._reload_engines(7)

    assert [name for name, _kwargs in calls] == [
        "pull_weights",
        "pause_generation",
        "flush_cache",
        "update_weights_from_disk",
        "continue_generation",
    ]
    assert calls[1][1] == {"mode": "retract"}
    assert calls[0][1] == {
        "target_version": 7,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }
    assert calls[3][1] == {"model_path": "/local/ckpt", "weight_version": "7"}


def test_in_place_pause_mode_skips_the_flush():
    # [FUNCTION DOC] With the pause mode set to in-place, reloading as rank zero must not issue a
    # flush cache call.
    """in_place pause mode does not flush."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)
    protocol.args.pause_generation_mode = "in_place"

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        protocol._reload_engines(7)

    assert "flush_cache" not in [name for name, _kwargs in calls]


def test_non_source_rank_issues_no_requests():
    # [FUNCTION DOC] Reloading as rank one (not the source rank) must issue no engine requests.
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 1
        protocol._reload_engines(7)

    assert calls == []


def _capture_baseline(protocol: UpdateWeightFromDiskDelta, tmp_path) -> None:
    # [FUNCTION DOC] HELPER: Prepares the protocol (delta dir under the temp path, a fake base
    # checkpoint path, empty snapshot, not a sender) and runs its baseline capture as rank zero in a
    # world of size one, with the distributed module, gloo group and tensor reader factory patched,
    # passing a weight source that yields no weights. Returns nothing; effects are the engine calls
    # logged by the protocol's fake engine.
    protocol.delta_dir = str(tmp_path / "delta")
    protocol.args.hf_checkpoint = "/fake/hf"
    protocol._snapshot = {}
    protocol.is_sender = False

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name, **kwargs: None),
    ):
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        protocol._capture_baseline(lambda materialize: [])


def test_baseline_capture_pulls_with_both_checkpoint_dirs(tmp_path):
    # [FUNCTION DOC] After baseline capture, asserts the engine calls are exactly pull weights then
    # get weight version, and the pull carries target version zero, the local checkpoint dir and the
    # shared source dir.
    """The baseline pull carries both checkpoint dirs too."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)

    _capture_baseline(protocol, tmp_path)

    assert [name for name, _kwargs in calls] == ["pull_weights", "get_weight_version"]
    assert calls[0][1] == {
        "target_version": 0,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }


def test_baseline_capture_reloads_the_pulled_checkpoint_when_equality_is_checked(tmp_path):
    # [FUNCTION DOC] With the equality check enabled, baseline capture issues pull weights followed
    # by update weights from disk (instead of get weight version); the reload uses the local
    # checkpoint dir as model path and the weight version string "0".
    """check_weight_update_equal makes the baseline reload the base checkpoint it just pulled."""
    calls: list[tuple[str, dict]] = []
    protocol = _make_protocol(calls)
    protocol.args.check_weight_update_equal = True

    _capture_baseline(protocol, tmp_path)

    assert [name for name, _kwargs in calls] == ["pull_weights", "update_weights_from_disk"]
    assert calls[1][1] == {"model_path": "/local/ckpt", "weight_version": "0"}


def test_non_source_rank_waits_for_baseline_engine_reload(tmp_path):
    # [FUNCTION DOC] Runs baseline capture as rank one in a world of size one (setup duplicated from
    # the capture helper) and asserts the patched distributed barrier was called exactly twice,
    # i.e. the non-source rank waits at barriers while rank zero does the engine work.
    """A non-source rank waits until rank zero finishes the baseline engine reload."""
    protocol = _make_protocol([])
    protocol.delta_dir = str(tmp_path / "delta")
    protocol.args.hf_checkpoint = "/fake/hf"
    protocol._snapshot = {}
    protocol.is_sender = False

    with (
        patch(f"{_MODULE}.dist") as dist_mock,
        patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda name, **kwargs: None),
    ):
        dist_mock.get_rank.return_value = 1
        dist_mock.get_world_size.return_value = 1
        protocol._capture_baseline(lambda materialize: [])

    assert dist_mock.barrier.call_count == 2
