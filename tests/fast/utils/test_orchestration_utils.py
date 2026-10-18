# [MODULE DOC]
# Tests for the orchestration script initializer, which sets up the shared driver machinery: restore
# the event logger checkpoint, configure the main-process logger, start the optional periodic
# stack dump, initialize tracking, launch the worker manager, and initialize the object store
# without contributing a memory segment; it returns the worker manager handle and registers
# teardown of the manager and tracking on a disposer. Tests patch each step with fakes to check
# order and arguments, that partially initialized tracking is released on failure, and that
# teardown happens on disposer exit in order (manager, then tracking).
from argparse import Namespace

import pytest

import miles.utils.orchestration_utils as orchestration_utils
from miles.utils.async_utils import Disposer
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.tracking_utils import tracking
from miles.utils.tracking_utils.base import TrackingManager


class TestInitOrchestrationScript:
    # [CLASS DOC] Tests of the orchestration script initializer.
    async def test_partial_tracking_initialization_releases_started_backends(
        self, monkeypatch: pytest.MonkeyPatch, partially_failing_tracking_manager: TrackingManager
    ) -> None:
        # [FUNCTION DOC] Install a tracking manager fixture whose later backend fails, stub the
        # other early steps, and run the initializer inside a disposer scope; expect a runtime
        # error ("tracking backend unavailable") and that the resource list the namespace carries
        # is empty afterwards (started backends were released).
        """A later backend failure still releases previously initialized tracking resources."""
        resources: list[object] = []
        monkeypatch.setattr(tracking, "_manager", partially_failing_tracking_manager)
        monkeypatch.setattr(orchestration_utils.event_logger_checkpoint, "restore", lambda _args: None)
        monkeypatch.setattr(orchestration_utils, "configure_logger", lambda _args, *, source: None)
        monkeypatch.setattr(orchestration_utils, "maybe_start_periodic_pyspy_dump", lambda: None)

        with pytest.raises(RuntimeError, match="tracking backend unavailable"):
            async with Disposer() as disposer:
                orchestration_utils.init_orchestration_script(
                    Namespace(enabled=True, resources=resources), disposer=disposer
                )

        assert resources == []

    def test_initializes_the_shared_driver_machinery_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Replace every step with a recording fake. Run the initializer and assert the
        # call order: checkpoint restore, logger configuration, periodic stack dump start, tracking
        # init, worker manager launch, object store init. Each step got the same arguments object;
        # the logger source is the main-component simple identity; the object store is told not to
        # contribute a segment; the result is the launched worker manager.
        """Every driver restores the event history first, then initializes the shared machinery in order."""
        args = Namespace(run="test")
        worker_manager = object()
        calls: list[str] = []
        captured: dict[str, object] = {}

        def fake_restore(actual_args: Namespace) -> None:
            # [FUNCTION DOC] HELPER: Record the call and capture the arguments.
            calls.append("event_logger_checkpoint.restore")
            captured["restore_args"] = actual_args

        def fake_configure_logger(actual_args: Namespace, *, source: SimpleProcessIdentity) -> None:
            # [FUNCTION DOC] HELPER: Record the call; capture the arguments and the process identity.
            calls.append("configure_logger")
            captured["logger_args"] = actual_args
            captured["source"] = source

        def fake_maybe_start_periodic_pyspy_dump() -> None:
            # [FUNCTION DOC] HELPER: Record the call.
            calls.append("maybe_start_periodic_pyspy_dump")

        def fake_init_tracking(actual_args: Namespace) -> None:
            # [FUNCTION DOC] HELPER: Record the call and capture the arguments.
            calls.append("init_tracking")
            captured["tracking_args"] = actual_args

        def fake_launch_worker_manager(actual_args: Namespace) -> object:
            # [FUNCTION DOC] HELPER: Record the call, capture the arguments and return the sentinel
            # worker manager object.
            calls.append("launch_worker_manager")
            captured["worker_manager_args"] = actual_args
            return worker_manager

        def fake_init_object_store(actual_args: Namespace, *, contribute_segment: bool) -> None:
            # [FUNCTION DOC] HELPER: Record the call; capture the arguments and the contribute flag.
            calls.append("object_store.init_instance")
            captured["object_store_args"] = actual_args
            captured["contribute_segment"] = contribute_segment

        monkeypatch.setattr(orchestration_utils.event_logger_checkpoint, "restore", fake_restore)
        monkeypatch.setattr(orchestration_utils, "configure_logger", fake_configure_logger)
        monkeypatch.setattr(
            orchestration_utils,
            "maybe_start_periodic_pyspy_dump",
            fake_maybe_start_periodic_pyspy_dump,
        )
        monkeypatch.setattr(orchestration_utils, "init_tracking", fake_init_tracking)
        monkeypatch.setattr(orchestration_utils, "launch_worker_manager", fake_launch_worker_manager)
        monkeypatch.setattr(orchestration_utils.object_store, "init_instance", fake_init_object_store)

        result = orchestration_utils.init_orchestration_script(args, disposer=Disposer())

        assert calls == [
            "event_logger_checkpoint.restore",
            "configure_logger",
            "maybe_start_periodic_pyspy_dump",
            "init_tracking",
            "launch_worker_manager",
            "object_store.init_instance",
        ]
        assert captured["restore_args"] is args
        assert captured["logger_args"] is args
        assert captured["source"] == SimpleProcessIdentity(component="main")
        assert captured["tracking_args"] is args
        assert captured["worker_manager_args"] is args
        assert captured["object_store_args"] is args
        assert captured["contribute_segment"] is False
        assert result is worker_manager

    async def test_the_manager_and_the_tracking_are_released_by_the_disposer_it_is_handed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Stub all steps; tracking finish and manager shutdown record to a shared
        # list. Inside the disposer scope nothing is released yet; after it exits the list is the
        # worker manager then "tracking".
        """Four drivers each remembering these two teardowns is four chances to forget one of them."""
        worker_manager = object()
        released: list[object] = []

        async def fake_shutdown_worker_manager(handle: object) -> None:
            # [FUNCTION DOC] HELPER: Record the handle being shut down.
            released.append(handle)

        monkeypatch.setattr(orchestration_utils.event_logger_checkpoint, "restore", lambda _args: None)
        monkeypatch.setattr(orchestration_utils, "configure_logger", lambda _args, *, source: None)
        monkeypatch.setattr(orchestration_utils, "maybe_start_periodic_pyspy_dump", lambda: None)
        monkeypatch.setattr(orchestration_utils, "init_tracking", lambda _args: None)
        monkeypatch.setattr(orchestration_utils, "launch_worker_manager", lambda _args: worker_manager)
        monkeypatch.setattr(
            orchestration_utils.object_store, "init_instance", lambda _args, *, contribute_segment: None
        )
        monkeypatch.setattr(orchestration_utils, "finish_tracking", lambda: released.append("tracking"))
        monkeypatch.setattr(orchestration_utils, "shutdown_worker_manager", fake_shutdown_worker_manager)

        async with Disposer() as disposer:
            orchestration_utils.init_orchestration_script(Namespace(run="test"), disposer=disposer)

            assert released == []

        assert released == [worker_manager, "tracking"]
