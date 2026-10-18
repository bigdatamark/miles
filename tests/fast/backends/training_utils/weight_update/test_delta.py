# [MODULE DOC]
# CPU tests for the disk-delta weight update protocol: that its optional post-write hook is
# resolved through the shared function registry at construction, and that the engine reload sequence
# (pull weights, pause generation, flush cache, reload from disk, continue generation) stops at the
# first engine call that reports failure. Engines are replaced by fake API clients that record calls.
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

_DELTA_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


class _RejectingApiClient:
    # [CLASS DOC] HELPER: Fake engine API client that accepts any method name. Every call is
    # recorded as a pair of method name and keyword arguments in a shared list; the one configured
    # failing method returns a failure result with an error message, all others return success.
    def __init__(self, calls: list[tuple[str, dict]], failing_method: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the shared call log and the name of the method to fail.
        self._calls = calls
        self._failing_method = failing_method

    def __getattr__(self, name: str):
        # [FUNCTION DOC] HELPER: Called for any undefined attribute; returns an async function for
        # that method name. The function logs the call and returns a mapping with success false and
        # the error message "engine rejected the weights" if the name is the failing one, otherwise
        # a mapping with success true.
        async def method(**kwargs):
            # [FUNCTION DOC] HELPER: The generated fake engine method; see the enclosing function.
            self._calls.append((name, kwargs))
            if name == self._failing_method:
                return {"success": False, "error_message": "engine rejected the weights"}
            return {"success": True}

        return method


class TestPostWriteHookConstruction:
    # [CLASS DOC] Tests of how the delta protocol constructor obtains its post-write hook.
    def test_configured_post_write_hook_is_loaded_from_function_registry(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Builds the protocol from args naming a disk directory under the temp path, xor
        # delta encoding, xxh3 checksum and a custom post-write function path, with the registry's
        # loader patched to return a sentinel object. Asserts the loader was called once with exactly
        # that path and the protocol stored the sentinel as its post-write hook.
        """A configured post-write path becomes the hook, resolved through the shared function registry."""
        hook = object()
        args = Namespace(
            update_weight_disk_dir=str(tmp_path / "delta"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3",
            custom_update_weight_post_write_path="miles_plugins.example:upload_delta",
        )

        with patch("miles.utils.function_registry.load_function", return_value=hook) as load_function:
            protocol = UpdateWeightFromDiskDelta(args)

        load_function.assert_called_once_with("miles_plugins.example:upload_delta")
        assert protocol._post_write_hook is hook


class TestReloadEnginesFailureTransitions:
    # [CLASS DOC] Tests that the protocol's engine reload aborts on a failing engine call.
    @staticmethod
    def _make_protocol(calls: list[tuple[str, dict]], failing_method: str) -> UpdateWeightFromDiskDelta:
        # [FUNCTION DOC] HELPER: Builds a protocol bypassing its constructor, with args for local
        # checkpoint dir "/local/ckpt", shared dir "/shared/delta", retract pause mode and no
        # equality check, one rejecting fake engine (logging into the given list and failing the
        # given method), no post-write hook and version dir "/shared/delta/v7". Returns it.
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(
            update_weight_local_checkpoint_dir="/local/ckpt",
            update_weight_disk_dir="/shared/delta",
            pause_generation_mode="retract",
            check_weight_update_equal=False,
        )
        protocol.rollout_engines = [_RejectingApiClient(calls, failing_method)]
        protocol._post_write_hook = None
        protocol._version_dir = "/shared/delta/v7"
        return protocol

    @pytest.mark.parametrize(
        ("failing_method", "expected_calls"),
        [
            ("pull_weights", ["pull_weights"]),
            (
                "update_weights_from_disk",
                ["pull_weights", "pause_generation", "flush_cache", "update_weights_from_disk"],
            ),
        ],
    )
    def test_reload_engine_failure_stops_before_the_next_lifecycle_phase(
        self, failing_method: str, expected_calls: list[str]
    ) -> None:
        # [FUNCTION DOC] Parametrized: if the pull fails, only the pull is issued; if the disk reload
        # fails, the calls are pull, pause generation, flush cache and the reload, and no continue
        # call follows. Runs the reload for version seven as rank zero (distributed module and gloo
        # group patched), expects a runtime error containing the engine's error message, and asserts
        # the recorded call names equal the expected list.
        """A rejected pull never pauses the engine, and a rejected disk reload never resumes it."""
        calls: list[tuple[str, dict]] = []
        protocol = self._make_protocol(calls, failing_method)

        with (
            patch(f"{_DELTA_MODULE}.dist") as dist_mock,
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
        ):
            dist_mock.get_rank.return_value = 0
            with pytest.raises(RuntimeError, match="engine rejected the weights"):
                protocol._reload_engines(7)

        assert [name for name, _kwargs in calls] == expected_calls
