# [MODULE DOC] Tests for the mock sglang engine actor: contract tests that keep its public method
# names and signatures aligned with the real command actor (so mock-driven tests cannot pass on
# calls that fail in production), smoke tests that run the mock as a real Ray actor in local mode
# (method round trip, call log ordering, one-shot fault injection), and tests of the port probe
# and node ip reporting (including the host ip environment override). Module constant: the set of
# method names that only the mock exposes for driving it from tests.
from __future__ import annotations

import inspect

import pytest
import ray

from miles.utils.http_utils import MILES_HOST_IP_ENV
from miles.utils.misc import NodeProbeMixin, get_free_port
from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine
from miles.utils.workers.command_actor import CommandActor


def _public_methods(cls) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of function member names of a class that are public
    # (no leading underscore) or are one of the underscored non-dunder names defined by the node
    # probe mixin; dunder names are always excluded. A class wrapped by Ray's remote decorator is
    # first unwrapped to the original class.
    """Public methods (and known semi-private helpers) of ``cls``."""
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__  # unwrap @ray.remote
    keep_underscored = {name for name in vars(NodeProbeMixin) if not name.startswith("__")}
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("__") and (not name.startswith("_") or name in keep_underscored)
    }


def _comparable_params(func) -> list[tuple[str, str, bool]]:
    # [FUNCTION DOC] HELPER: Returns, for each parameter of the function's signature (dropping a
    # leading self and the Ray-injected trace context parameter), a triple of the parameter name,
    # the parameter kind name, and whether it has a default value.
    """Parameter name, kind and default-ness of ``func``, dropping a leading ``self`` and
    the ``_ray_trace_ctx`` argument that ``ray.remote`` injects."""
    params = list(inspect.signature(func).parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return [
        (p.name, p.kind.name, p.default is not inspect.Parameter.empty) for p in params if p.name != "_ray_trace_ctx"
    ]


# Helpers the mock exposes for tests to drive it; the real actor has no reason to carry them.
_MOCK_ONLY_METHODS: set[str] = {
    "get_calls",
    "get_http_paths",
    "get_http_payloads_of",
    "get_server_args",
    "set_fault",
}


# ----------------------------- contract tests -----------------------------


class TestApiContractMatchesRealEngine:
    # [CLASS DOC] Contract tests comparing the mock engine class with the real command actor class
    # by reflection (no actor is started).
    def test_the_mock_answers_every_call_the_real_actor_answers(self) -> None:
        # [FUNCTION DOC] Asserts every public method of the real actor also exists on the mock;
        # the failure message lists the missing names.
        """A method only the real actor has would AttributeError in production while
        every mock-driven test stays green."""
        missing = _public_methods(CommandActor) - _public_methods(MockSGLangEngine)
        assert missing == set(), (
            f"MockSGLangEngine cannot stand in for CommandActor: {sorted(missing)}. "
            f"Add stub implementations to mock_sglang_engine.py."
        )

    def test_the_mock_invents_no_method_the_real_actor_lacks(self) -> None:
        # [FUNCTION DOC] Asserts every public method of the mock exists on the real actor or is in
        # the mock-only allowance set.
        """A stub with no counterpart lets a mock-driven test go green on a call that
        AttributeErrors in production."""
        invented = _public_methods(MockSGLangEngine) - _public_methods(CommandActor) - _MOCK_ONLY_METHODS
        assert invented == set(), (
            f"MockSGLangEngine exposes methods CommandActor does not: {sorted(invented)}. "
            f"Either add them to CommandActor or list them in _MOCK_ONLY_METHODS."
        )

    def test_signature_compat_for_run(self) -> None:
        # [FUNCTION DOC] Asserts the run method's parameter name sets (excluding self and the Ray
        # trace context) are equal between the real actor and the unwrapped mock class. Kinds and
        # defaults are not compared here.
        """``run`` is the most important signature to keep aligned because
        the rollout code hands it the launch command and env map."""
        real_sig = inspect.signature(CommandActor.run)
        mock_sig = inspect.signature(MockSGLangEngine.__ray_actor_class__.run)
        # ray injects _ray_trace_ctx into the methods of a class it decorates.
        ignored = {"self", "_ray_trace_ctx"}
        assert set(real_sig.parameters) - ignored == set(mock_sig.parameters) - ignored

    def test_signature_compat_for_every_shared_method(self) -> None:
        # [FUNCTION DOC] For each method present on both classes (in sorted order), skips those
        # whose mock version accepts arbitrary keyword arguments, and otherwise compares the
        # comparable parameter lists; collects mismatches by method name and asserts none, with a
        # message listing them.
        """Every method shared by mock and real engine declares the same parameter names, kinds and default-ness."""
        mock_cls = MockSGLangEngine.__ray_actor_class__
        shared: set[str] = _public_methods(CommandActor) & _public_methods(MockSGLangEngine)

        mismatches: dict[str, dict[str, list[tuple[str, str, bool]]]] = {}
        for name in sorted(shared):
            mock_params = _comparable_params(getattr(mock_cls, name))
            if any(kind == "VAR_KEYWORD" for _name, kind, _has_default in mock_params):
                continue
            real_params = _comparable_params(getattr(CommandActor, name))
            if mock_params != real_params:
                mismatches[name] = {"mock": mock_params, "real": real_params}

        assert not mismatches, (
            f"MockSGLangEngine signatures drifted from CommandActor: {mismatches}. "
            f"Update mock_sglang_engine.py so the mock accepts exactly what the real engine accepts."
        )


class TestNodeProbeHelpersAreCovered:
    # [CLASS DOC] Guards that the node probe mixin helpers are part of the contract comparison.
    def test_every_node_probe_helper_takes_part_in_the_contract_comparison(self) -> None:
        # [FUNCTION DOC] Asserts the mixin defines at least one non-dunder member and that all of
        # them are included in the public method sets of both the real actor and the mock.
        """A probe helper left out of the comparison set lets the mock silently lose a stub the real actor has."""
        probe_helpers = {name for name in vars(NodeProbeMixin) if not name.startswith("__")}

        assert probe_helpers
        assert probe_helpers <= _public_methods(CommandActor)
        assert probe_helpers <= _public_methods(MockSGLangEngine)


# ----------------------------- real Ray smoke tests -----------------------------


class TestRealRayActorLifecycle:
    # [CLASS DOC] Smoke tests that start the mock as a real Ray actor (requesting a tenth of a
    # CPU and no GPU) using the local-mode Ray fixture provided elsewhere in the test suite.
    def test_actor_construction_and_method_round_trip(self, ray_local_mode):
        # [FUNCTION DOC] Creates the actor, picks a free port from 20000, and calls run with a
        # fake sglang launch command (with that port and an empty env map), a free port block probe
        # from port 20100 with count 1, and kill subprocess. Asserts the call log lists exactly
        # those three method names in order and that the recorded server arguments have host
        # 127.0.0.1 and the chosen port. Always shuts the actor down and kills it in a nested
        # cleanup.
        """End-to-end: every method rollout code touches round-trips through
        Ray with the right args, and the call log preserves ordering."""
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote()
        try:
            port = get_free_port(start_port=20000)
            cmd = f"python -m sglang.launch_server --model-path /fake/model --host 127.0.0.1 --port {port}"
            ray.get(actor.run.remote(cmd=cmd, envs={}))
            ray.get(actor._get_free_port_block.remote(start_port=20100, count=1))
            ray.get(actor.kill_subprocess.remote())

            calls = ray.get(actor.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert method_names == [
                "run",
                "_get_free_port_block",
                "kill_subprocess",
            ]
            server_args = ray.get(actor.get_server_args.remote())
            assert server_args["host"] == "127.0.0.1" and server_args["port"] == port
        finally:
            try:
                ray.get(actor.shutdown.remote())
            finally:
                ray.kill(actor)

    def test_fault_injection_round_trips_through_ray(self, ray_local_mode):
        # [FUNCTION DOC] Schedules a runtime error with message boom for the shutdown method, then
        # asserts the first shutdown call raises a Ray task error matching boom and that a second
        # shutdown call succeeds and returns true (the fault is one-shot). Kills the actor at the
        # end.
        """``set_fault`` schedules an exception; it must surface back via
        ``ray.get`` and be one-shot (cleared after firing)."""
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote()
        try:
            ray.get(actor.set_fault.remote("shutdown", RuntimeError("boom")))
            with pytest.raises(ray.exceptions.RayTaskError, match="boom"):
                ray.get(actor.shutdown.remote())
            # Fault is one-shot — second call must succeed.
            assert ray.get(actor.shutdown.remote()) is True
        finally:
            ray.kill(actor)


class TestPortProbe:
    # [CLASS DOC] Tests of the mock's port availability probe, driven in process on the
    # unwrapped class (no Ray).
    def test_the_mock_reports_every_port_as_free_and_records_the_probe(self) -> None:
        # [FUNCTION DOC] Probing port 30001 returns true and the call log holds exactly that one
        # call with its keyword argument.
        """A mock-driven launch must never be refused a pinned port by the real machine's port state."""
        engine = MockSGLangEngine.__ray_actor_class__()

        assert engine._is_port_available(port=30001) is True
        assert engine.get_calls() == [("_is_port_available", (), {"port": 30001})]


class TestNodeAddress:
    # [CLASS DOC] Tests that the mock reports a routable node address.
    def test_the_mock_reports_the_node_ip_instead_of_loopback(self) -> None:
        # [FUNCTION DOC] The mock's node ip equals the mixin's node ip and is not 127.0.0.1.
        """A mock engine placed on another node must publish an address its peers can reach."""
        engine = MockSGLangEngine.__ray_actor_class__()

        assert engine._get_node_ip() == NodeProbeMixin._get_node_ip()
        assert engine._get_node_ip() != "127.0.0.1"


class TestNodeIpReporting:
    # [CLASS DOC] Tests of the node ip method of the mock, including call recording and the host
    # ip environment override.
    def test_get_node_ip_reports_the_reachable_node_ip_and_records_the_call(self) -> None:
        # [FUNCTION DOC] The node ip equals the mixin's value, is not loopback, and the call log
        # holds exactly one entry for that method with no arguments. (Overlaps with the test in
        # the node address class.)
        """Driven in process, the mock hands back the node's routable ip rather than loopback."""
        engine = MockSGLangEngine.__ray_actor_class__()

        node_ip = engine._get_node_ip()

        assert node_ip == NodeProbeMixin._get_node_ip()
        assert node_ip != "127.0.0.1"
        assert engine.get_calls() == [("_get_node_ip", (), {})]

    def test_the_mock_honours_the_node_ip_override(self, monkeypatch) -> None:
        # [FUNCTION DOC] With the host ip environment variable set to 10.9.9.9, the mock reports
        # that address.
        """A real worker publishes MILES_HOST_IP when it is set, so a mock that ignores it tests the wrong address."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.9.9.9")

        assert MockSGLangEngine.__ray_actor_class__()._get_node_ip() == "10.9.9.9"

    def test_the_mock_and_the_real_actor_report_the_same_node_ip(self, monkeypatch) -> None:
        # [FUNCTION DOC] With the override set to 10.9.9.9, a directly constructed real command
        # actor and the mock report the same node ip.
        """Tests drive the mock in place of the real actor, so the two must derive an address the same way."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.9.9.9")

        assert MockSGLangEngine.__ray_actor_class__()._get_node_ip() == CommandActor()._get_node_ip()
