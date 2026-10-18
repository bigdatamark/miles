# [MODULE DOC] Unit tests for how the prometheus utilities construct their collector actor: the
# Ray module inside the utilities is replaced by a fake whose actor class records the arguments
# of the remote constructor call, and the tests assert the arguments are passed by keyword, the
# collector's constructor parameters are all keyword-only, and the recorded keywords bind to the
# real constructor signature.
import inspect
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
from tests.fast.utils.fake_ray_ids import fake_ray_node_id

from miles.utils.tracking_utils import prometheus_utils


class TestCollectorConstruction:
    # [CLASS DOC] Test of the construction call made by the init function.
    def test_the_collector_actor_is_constructed_with_keyword_arguments(self, monkeypatch):
        # [FUNCTION DOC] Replaces the module's ray handle (reverted at teardown) with a fake that
        # wraps the class in a recording actor class (options returns itself, remote records
        # positional and keyword arguments and returns a handle with a ping), and clears the
        # module's cached collector handle. Calls the init with the server start flag and asserts
        # no positional arguments and exactly the keyword argument args with the namespace.
        """A positional handoff silently binds to the wrong parameter once the actor grows another one."""
        recorded: dict[str, object] = {}
        args = SimpleNamespace(prometheus_run_name="run")

        class _FakeActorClass:
            # [CLASS DOC] HELPER: Fake actor class: options is a no-op returning itself; remote
            # records its arguments and returns a stub handle.
            def options(self, **_options):
                # [FUNCTION DOC] HELPER: Ignores the options and returns the same object.
                return self

            def remote(self, *call_args, **call_kwargs):
                # [FUNCTION DOC] HELPER: Stores positional and keyword arguments in the outer
                # record and returns a stub handle whose ping remote call returns nothing.
                recorded["call_args"] = call_args
                recorded["call_kwargs"] = call_kwargs
                return SimpleNamespace(ping=SimpleNamespace(remote=lambda: None))

        monkeypatch.setattr(
            prometheus_utils,
            "ray",
            SimpleNamespace(
                remote=lambda _cls: _FakeActorClass(),
                get=lambda _ref: None,
                get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: fake_ray_node_id(0)),
            ),
        )
        monkeypatch.setattr(prometheus_utils, "_collector_handle", None)

        prometheus_utils.init_prometheus(args, start_server=True)

        assert (recorded["call_args"], recorded["call_kwargs"]) == ((), {"args": args})


class _CollectorInit(NamedTuple):
    # [CLASS DOC] HELPER: Record of the collector construction: the class that was wrapped by the
    # fake remote decorator, and the positional and keyword arguments of the remote call.
    actor_class: type
    call_args: tuple
    call_kwargs: dict


def _run_init_prometheus(monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace) -> _CollectorInit:
    # [FUNCTION DOC] HELPER: Patches the module's ray handle with a fake (remote decorator that
    # records the class, a get that returns nothing, a runtime context with a fixed fake node id)
    # and clears the cached collector handle; calls the init with the server start flag; returns
    # the recorded class and call arguments. Patches are reverted at teardown.
    captured: dict[str, Any] = {}

    class _FakeActorClass:
        # [CLASS DOC] HELPER: Fake actor class recording remote call arguments.
        def options(self, **_options: Any) -> "_FakeActorClass":
            # [FUNCTION DOC] HELPER: Ignores the options and returns itself.
            return self

        def remote(self, *call_args: Any, **call_kwargs: Any) -> SimpleNamespace:
            # [FUNCTION DOC] HELPER: Records the arguments and returns a stub handle with a ping.
            captured["call_args"] = call_args
            captured["call_kwargs"] = call_kwargs
            return SimpleNamespace(ping=SimpleNamespace(remote=lambda: None))

    def _fake_remote(cls: type) -> _FakeActorClass:
        # [FUNCTION DOC] HELPER: Records the wrapped class and returns a fake actor class.
        captured["actor_class"] = cls
        return _FakeActorClass()

    monkeypatch.setattr(
        prometheus_utils,
        "ray",
        SimpleNamespace(
            remote=_fake_remote,
            get=lambda _ref: None,
            get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: fake_ray_node_id(0)),
        ),
    )
    monkeypatch.setattr(prometheus_utils, "_collector_handle", None)

    prometheus_utils.init_prometheus(args, start_server=True)

    return _CollectorInit(
        actor_class=captured["actor_class"],
        call_args=captured["call_args"],
        call_kwargs=captured["call_kwargs"],
    )


class TestCollectorKeywordOnlyConstruction:
    # [CLASS DOC] Tests that the collector class takes only keyword arguments.
    def test_the_collector_refuses_a_positional_args_object(self):
        # [FUNCTION DOC] Constructing the collector with a positional args namespace raises a
        # type error.
        """Constructing the collector positionally must fail so a later parameter cannot silently steal the slot."""
        with pytest.raises(TypeError):
            prometheus_utils._PrometheusCollector(SimpleNamespace(prometheus_port=0))

    def test_every_collector_constructor_parameter_is_keyword_only(self):
        # [FUNCTION DOC] Inspects the constructor signature (skipping self): every parameter is
        # keyword-only, and there is at least one.
        """No collector constructor parameter may be positionally bindable."""
        parameters = list(inspect.signature(prometheus_utils._PrometheusCollector.__init__).parameters.values())[1:]

        assert [parameter.kind for parameter in parameters] == [inspect.Parameter.KEYWORD_ONLY] * len(parameters)
        assert parameters

    def test_the_recorded_collector_keywords_bind_to_the_collector_constructor(self, monkeypatch):
        # [FUNCTION DOC] Runs the init with the fake ray; asserts the wrapped class is the real
        # collector class, no positional arguments were sent, and the recorded keyword arguments
        # bind to the constructor signature without error.
        """The keywords the call site sends must name real collector constructor parameters."""
        init = _run_init_prometheus(monkeypatch, SimpleNamespace(prometheus_run_name="run"))

        assert init.actor_class is prometheus_utils._PrometheusCollector
        assert init.call_args == ()
        inspect.signature(init.actor_class.__init__).bind(object(), **init.call_kwargs)
