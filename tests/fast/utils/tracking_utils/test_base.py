# [MODULE DOC] Unit tests for the tracking manager's backend gating: a backend is built only when
# its enable flag is set and its support check passes (the prometheus backend is supported only
# under the ray cluster backend), unsupported backends receive no metrics, the real registry
# entry for prometheus is gated, and a disabled backend is never asked about support. Test
# backends record their init arguments and logged metrics in class level lists.
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.tracking_utils.base import PrometheusBackend, TrackingBackend, TrackingManager
from miles.utils.tracking_utils.tracking import BACKEND_REGISTRY
from miles.utils.workers.types import ClusterBackend


class _RecordingBackend(TrackingBackend):
    # [CLASS DOC] HELPER: Backend that records init arguments and logged metric mappings in class
    # level lists (each subclass redefines its own lists); finish does nothing.
    inits: list[Any] = []
    logs: list[dict[str, Any]] = []

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Appends the args to the class level init list.
        type(self).inits.append(args)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Appends the metrics mapping to the class level log list.
        type(self).logs.append(metrics)

    def finish(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return


class _RayOnlyBackend(_RecordingBackend):
    # [CLASS DOC] HELPER: Recording backend whose support check always says unsupported.
    inits: list[Any] = []
    logs: list[dict[str, Any]] = []

    @classmethod
    def is_supported(cls, args) -> bool:
        # [FUNCTION DOC] HELPER: Always returns false.
        return False


class _PickyBackend(_RecordingBackend):
    # [CLASS DOC] HELPER: Recording backend whose support check fails the test if consulted.
    inits: list[Any] = []
    logs: list[dict[str, Any]] = []

    @classmethod
    def is_supported(cls, args) -> bool:
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying support must not be
        # consulted for a disabled backend.
        raise AssertionError("is_supported must not be consulted for a disabled backend")


class TestPrometheusBackendIsSupported:
    # [CLASS DOC] Tests of the real prometheus backend's support check.
    @pytest.mark.parametrize(
        ("cluster_backend", "expected"), [(ClusterBackend.RAY, True), (ClusterBackend.KUBERNETES, False)]
    )
    def test_follows_the_cluster_backend(self, cluster_backend, expected):
        # [FUNCTION DOC] Parametrized: supported (true) for the ray cluster backend and
        # unsupported (false) for kubernetes, given an args namespace carrying the backend value.
        """The collector is a named Ray actor, so it exists only under the ray backend."""
        assert PrometheusBackend.is_supported(SimpleNamespace(cluster_backend=cluster_backend.value)) is expected


class TestTrackingManagerInit:
    # [CLASS DOC] Tests of the manager's init, which takes a mapping from name to a backend class
    # and the name of the enabling argument flag.
    def setup_method(self):
        # [FUNCTION DOC] HELPER: Resets the recorded init and log lists of the three test backends
        # before each test.
        for cls in (_RecordingBackend, _RayOnlyBackend, _PickyBackend):
            cls.inits = []
            cls.logs = []

    def test_initialises_an_enabled_supported_backend(self):
        # [FUNCTION DOC] An enabled supported backend is initialised exactly once with the args.
        """The ordinary path is untouched: an enabled backend still gets built."""
        manager = TrackingManager({"recording": (_RecordingBackend, "use_recording")})
        args = SimpleNamespace(use_recording=True)

        manager.init(args)

        assert _RecordingBackend.inits == [args]

    def test_skips_an_enabled_unsupported_backend(self, caplog):
        # [FUNCTION DOC] An enabled but unsupported backend is not initialised, and a warning
        # mentioning its name is logged.
        """An enabled but unsupported backend is dropped whole, not half-initialised."""
        manager = TrackingManager({"ray_only": (_RayOnlyBackend, "use_ray_only")})

        with caplog.at_level(logging.WARNING):
            manager.init(SimpleNamespace(use_ray_only=True))

        assert _RayOnlyBackend.inits == []
        assert "ray_only" in caplog.text

    def test_unsupported_backend_receives_no_metrics(self):
        # [FUNCTION DOC] With one unsupported and one supported backend both enabled, logging a
        # loss metric reaches only the supported one.
        """Dropping the backend also removes it from the later log fan-out."""
        manager = TrackingManager(
            {
                "ray_only": (_RayOnlyBackend, "use_ray_only"),
                "recording": (_RecordingBackend, "use_recording"),
            }
        )
        manager.init(SimpleNamespace(use_ray_only=True, use_recording=True))

        manager.log({"loss": 1.0}, step=0)

        assert _RecordingBackend.logs == [{"loss": 1.0}]
        assert _RayOnlyBackend.logs == []

    def test_registry_gates_the_real_prometheus_entry(self):
        # [FUNCTION DOC] Using the real registry entry with prometheus enabled under the
        # kubernetes cluster backend, the manager ends up with no backends (checked through its
        # private backend list).
        """The registry entry itself is gated, not only a synthetic backend."""
        manager = TrackingManager({"prometheus": BACKEND_REGISTRY["prometheus"]})

        manager.init(SimpleNamespace(use_prometheus=True, cluster_backend=ClusterBackend.KUBERNETES.value))

        assert manager._backends == []

    def test_disabled_backend_is_never_asked_for_support(self):
        # [FUNCTION DOC] A disabled picky backend is not initialised and its support check (which
        # would raise) is never called.
        """The enable flag decides first, so a support check may read backend-only args."""
        manager = TrackingManager({"picky": (_PickyBackend, "use_picky")})

        manager.init(SimpleNamespace(use_picky=False))

        assert _PickyBackend.inits == []
