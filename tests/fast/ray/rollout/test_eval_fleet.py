# [MODULE DOC]
# Tests of the eval fleet: the inference-controller-side fleet (pins every engine of the eval server
# to a snapshot checkpoint over http, verifies the served weight version on all engines, retries once,
# and degrades to an attributable skip reason) and the rollout-executor-side fleet (asks the inference
# controller to pin over rpc, caches one generate state against the fleet router, and maps controller
# failures to skips). Registered as a CPU CI test in the stage-a-cpu suite with an estimated time of 60
# seconds. Uses fake engines, a fake eval server, fake controller and provider objects, and fixtures
# that neutralize or record the router readiness wait.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")


import asyncio
import inspect
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args as _make_args

import miles.ray.rollout.eval_fleet as eval_fleet_mod
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.eval_fleet import (
    EvalFleetInfo,
    EvalFleetPin,
    InferenceControllerEvalFleet,
    RolloutExecutorEvalFleet,
)
from miles.ray.rollout.rollout_server import RolloutServer
from miles.rollout.checkpoint_eval import EvalSkip
from miles.utils.context_lock import ContextLock
from miles.utils.workers.rpc.client.misc import RpcWorkerCallError, ServerRestartedError
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_spec import HostAndPort


def make_args(**overrides):
    # [FUNCTION DOC] HELPER: Wraps the conftest args builder with defaults of one eval gpu, one eval
    # gpu per engine, and a router table mapping default to 10.0.0.1 port 30000 and eval to 10.0.0.2
    # port 31000; explicit overrides win over these defaults.
    defaults = dict(
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
        sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)},
    )
    defaults.update(overrides)
    return _make_args(**defaults)


class FakeEngine:
    # [CLASS DOC] HELPER: Fake api client of one eval cell. Records each call into a shared log as a
    # triple of method name, positional arguments and keyword arguments, and remembers the last
    # weight version it was loaded with.
    """Stands in for the api client of one eval cell, with the two methods a pin calls."""

    def __init__(self, log):
        # [FUNCTION DOC] HELPER: Stores the shared log and starts with no weight version.
        self.log = log
        self.weight_version = None

    async def update_weights_from_disk(self, model_path, load_format=None, weight_version=None):
        # [FUNCTION DOC] HELPER: Logs a weight load with the model path and the weight version, then
        # stores the weight version as the engine's current one. KNOWN ISSUE: the load format
        # argument is accepted but ignored.
        self.log.append(("update_weights_from_disk", (model_path,), dict(weight_version=weight_version)))
        self.weight_version = weight_version

    async def get_weight_version(self):
        # [FUNCTION DOC] HELPER: Logs the query and returns the stored weight version (none if never
        # loaded).
        self.log.append(("get_weight_version", (), {}))
        return self.weight_version


class FakeEvalServer:
    # [CLASS DOC] HELPER: Fake eval rollout server exposing only what the fleet reads: the api clients
    # of its engines (guarded by the context lock), the lock itself and the router address (10.0.0.2
    # port 31000 by default, reassignable by tests). The test of the real-server interface guards that
    # everything offered here also exists on the real server class.
    def __init__(self, engines):
        # [FUNCTION DOC] HELPER: Stores the engine list, creates a context lock named for the fake
        # server, and sets the router ip and port.
        self._engines = engines
        self.context_lock = ContextLock("FakeEvalServer")
        self.router_ip = "10.0.0.2"
        self.router_port = 31000

    @property
    def api_clients(self):
        # [FUNCTION DOC] HELPER: Property. Asserts the context lock is held in the current context
        # (reading without the lock is a bug), then returns a fresh list copy of the current engines, so
        # membership changes made by tests are seen on the next read.
        assert self.context_lock.held_in_current_context, "api_clients is read under the server's lock"
        return list(self._engines)


class FlakyEngine(FakeEngine):
    # [CLASS DOC] HELPER: Fake engine whose weight load raises a runtime error for a configured number of
    # calls before behaving like the base fake.
    def __init__(self, log, *, failures: int):
        # [FUNCTION DOC] HELPER: Initializes the base fake and sets the count of failures still to emit.
        super().__init__(log)
        self.remaining_failures = failures

    async def update_weights_from_disk(self, model_path, load_format=None, weight_version=None):
        # [FUNCTION DOC] HELPER: While failures remain, decrements the counter and raises a runtime
        # error about a transient http failure (nothing is logged for failed calls); afterwards
        # delegates to the base fake.
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("transient http failure")
        await super().update_weights_from_disk(model_path, load_format=load_format, weight_version=weight_version)


class HangingEngine(FakeEngine):
    # [CLASS DOC] HELPER: Fake engine whose weight load never completes.
    async def update_weights_from_disk(self, model_path, load_format=None, weight_version=None):
        # [FUNCTION DOC] HELPER: Logs the load call, then waits forever on an event nobody sets (the
        # weight version is never stored).
        self.log.append(("update_weights_from_disk", (model_path,), dict(weight_version=weight_version)))
        await asyncio.Event().wait()


@pytest.fixture
def router_probes(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Replaces the fleet module's wait-for-http-ok function with an
    # async recorder that appends a record of url, json payload and timeout to a list (and does no
    # network I/O). Returns the list.
    probes: list[SimpleNamespace] = []

    async def record_probe(url, *, json_payload=None, timeout=180.0):
        # [FUNCTION DOC] HELPER: Appends a record of the probe arguments to the fixture's list.
        probes.append(SimpleNamespace(url=url, json_payload=json_payload, timeout=timeout))

    monkeypatch.setattr(eval_fleet_mod, "wait_http_ok", record_probe)
    return probes


@pytest.fixture
def router_ready(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Replaces the controller-side fleet's router readiness wait with
    # an async function that returns immediately, so pin tests do not probe a router. Returns nothing.
    async def noop_router_ready(self, timeout=180.0):
        # [FUNCTION DOC] HELPER: Does nothing and returns no value.
        return None

    monkeypatch.setattr(eval_fleet_mod.InferenceControllerEvalFleet, "_wait_router_ready", noop_router_ready)


def make_fleet(args, engines):
    # [FUNCTION DOC] HELPER: Builds a controller-side eval fleet over a fake eval server holding the
    # given engines.
    return InferenceControllerEvalFleet(args, srv=FakeEvalServer(engines))


class TestEvalFleetInfo:
    # [CLASS DOC] Checks the fleet description published by the controller-side fleet.
    def test_describes_the_fleet_its_router_serves(self):
        # [FUNCTION DOC] Builds a fleet with four eval gpus and two gpus per engine over a fake server
        # with no engines, and asserts its info equals the server's router address (10.0.0.2 port 31000)
        # with the args' gpu count and gpus per engine.
        """The description the executor retargets its eval args to comes from the server, not its own args."""
        fleet = make_fleet(make_args(eval_num_gpus=4, eval_num_gpus_per_engine=2), [])

        assert fleet.info == EvalFleetInfo(
            router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=4, num_gpus_per_engine=2
        )


def _answers_version(version: str):
    # [FUNCTION DOC] HELPER: Returns an async function (no arguments) that always answers the given
    # weight version string, for substituting an engine's weight version query.
    async def get_weight_version():
        # [FUNCTION DOC] HELPER: Returns the captured version string.
        return version

    return get_weight_version


class TestTheFleetIsBuiltOnTheRealServer:
    # [CLASS DOC] Guards the fake eval server against drifting away from the real rollout server.
    def test_everything_the_fake_server_offers_a_real_one_offers_too(self):
        # [FUNCTION DOC] Collects every public (not underscore-prefixed) instance attribute and class
        # attribute name of a fake server and asserts all of them are among the attribute names or
        # annotations of the real rollout server class.
        """A fake with an interface of its own is how the pin path came to call `.engines`, which no server has."""
        fake = FakeEvalServer([])
        offered = {name for name in (*vars(fake), *type(fake).__dict__) if not name.startswith("_")}

        assert offered <= set(dir(RolloutServer)) | set(RolloutServer.__annotations__)


class TestEvalFleetPinning:
    # [CLASS DOC] Checks the controller-side pin: weights are loaded on every engine, every engine must
    # report the pinned version, and failures become a skip reason in the returned pin. All tests but
    # one neutralize the router readiness wait.
    async def test_pins_every_engine_before_reporting_success(self, router_ready):
        # [FUNCTION DOC] Pins snapshot step 5 as version 5 over two engines; asserts two weight load
        # calls, each with weight version 5, and a pin result with no skip reason.
        """Every engine is reloaded from the snapshot before the pin reports no skip."""
        log = []
        fleet = make_fleet(make_args(), [FakeEngine(log), FakeEngine(log)])

        pin = await fleet.pin("/snap/step_5", "5")

        load_events = [e for e in log if e[0] == "update_weights_from_disk"]
        assert len(load_events) == 2
        assert all(e[2]["weight_version"] == "5" for e in load_events)
        assert pin == EvalFleetPin(skip_reason=None)

    async def test_requires_all_engines_to_match_and_retries(self, router_ready):
        # [FUNCTION DOC] One engine is good and the other always reports version 999. The pin must fail
        # with skip reason pin violation (even though one engine matches) after retrying once, so the
        # log shows four weight load calls (two engines times two attempts).
        """The router load-balances across engines, so one stale engine = mixed
        versions: the pin must fail even when the other engine matches, retry once,
        then degrade to an attributable skip."""
        log = []
        good, stale = FakeEngine(log), FakeEngine(log)
        stale.get_weight_version = _answers_version("999")
        fleet = make_fleet(make_args(), [good, stale])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"
        assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 4  # 2 engines x 2 attempts

    async def test_a_router_that_never_becomes_ready_skips_the_eval(self, monkeypatch):
        # [FUNCTION DOC] Replaces the router readiness wait with one that raises a timeout error and
        # asserts the pin returns skip reason unhealthy instead of raising.
        """Every other case here neutralises the router wait, so this branch is the only thing that reads it."""

        async def never_ready(self, timeout=180.0):
            # [FUNCTION DOC] HELPER: Always raises a timeout error saying the router never came up.
            raise TimeoutError("router never came up")

        monkeypatch.setattr(eval_fleet_mod.InferenceControllerEvalFleet, "_wait_router_ready", never_ready)
        fleet = make_fleet(make_args(), [FakeEngine([])])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "unhealthy"

    async def test_a_cell_that_joined_between_attempts_is_pinned_too(self, router_ready):
        # [FUNCTION DOC] The server starts with one stale engine (reports 999). The stale engine's weight
        # load is replaced by a function that, on the first attempt, swaps the server membership to a
        # new joined engine and makes the stale one report 5. The pin must succeed with no skip reason,
        # and the joined engine must hold weight version 5, proving membership is re-read per attempt.
        """Membership is read again per attempt, or a cell that joined mid-pin would serve the old weights."""
        log = []
        joined, stale = FakeEngine(log), FakeEngine(log)
        stale.get_weight_version = _answers_version("999")
        server = FakeEvalServer([stale])
        fleet = InferenceControllerEvalFleet(make_args(), srv=server)

        async def join_after_first_attempt(*_args, **_kwargs):
            # [FUNCTION DOC] HELPER: Replaces the server's engine list with only the joined engine and
            # makes the stale engine answer version 5; takes and ignores any arguments.
            server._engines = [joined]
            stale.get_weight_version = _answers_version("5")

        stale.update_weights_from_disk = join_after_first_attempt

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)
        assert joined.weight_version == "5"

    async def test_does_not_health_probe_the_server(self, router_ready):
        # [FUNCTION DOC] Asserts the fake server has none of the fault-tolerance methods (probe and
        # mark dead, recover, wait for all engines alive), then pins and asserts success with no skip
        # reason, i.e. the pin goes straight to the weight load.
        """The eval fleet has no fault tolerance: pin goes straight to the weight load."""
        server = FakeEvalServer([FakeEngine([])])
        assert not any(hasattr(server, name) for name in ("probe_and_mark_dead", "recover", "wait_all_engines_alive"))

        pin = await InferenceControllerEvalFleet(make_args(), srv=server).pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)


class TestPinWeightTransport:
    # [CLASS DOC] Checks how the pin talks to engines: which arguments are sent and that they fit the
    # real api client.
    async def test_pin_sends_the_checkpoint_dir_as_the_http_model_path(self, router_ready):
        # [FUNCTION DOC] Pins snapshot step 7 as version 7 over two engines and asserts the weight load
        # calls in the log are exactly two, each with the snapshot directory as the model path and
        # weight version 7.
        """The snapshot directory reaches every engine as the model path of the weight load, not as any other field."""
        log = []
        fleet = make_fleet(make_args(), [FakeEngine(log), FakeEngine(log)])

        await fleet.pin("/snap/step_7", "7")

        assert [e for e in log if e[0] == "update_weights_from_disk"] == [
            ("update_weights_from_disk", ("/snap/step_7",), dict(weight_version="7")),
            ("update_weights_from_disk", ("/snap/step_7",), dict(weight_version="7")),
        ]

    async def test_the_calls_pin_makes_bind_against_the_real_api_client(self):
        # [FUNCTION DOC] Creates a real sglang api client for the eval router url (no request is sent)
        # and checks by signature binding that the weight load accepts a model path and weight version
        # keyword, and the weight version query accepts no extra arguments. Also asserts both are
        # coroutine functions.
        """The fleet talks HTTP now, so the two calls it makes must stay bindable against the real client."""
        client = SGLangApiClient(server_url="http://10.0.0.2:31000")

        inspect.signature(SGLangApiClient.update_weights_from_disk).bind(
            client, model_path="/snap/step_7", weight_version="7"
        )
        inspect.signature(SGLangApiClient.get_weight_version).bind(client)

        assert inspect.iscoroutinefunction(SGLangApiClient.update_weights_from_disk)
        assert inspect.iscoroutinefunction(SGLangApiClient.get_weight_version)


class TestPinFailureModes:
    # [CLASS DOC] Checks that load failures, hangs and an empty fleet degrade to skip reasons.
    async def test_pin_retries_a_transient_load_failure_and_then_succeeds(self, router_ready):
        # [FUNCTION DOC] One healthy engine plus one that fails its first load. The pin must succeed
        # with no skip reason and the flaky engine's failure budget must be fully consumed.
        """One engine failing its first weight load is retried, and the eval point still runs."""
        log = []
        flaky = FlakyEngine(log, failures=1)
        fleet = make_fleet(make_args(), [FakeEngine(log), flaky])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)
        assert flaky.remaining_failures == 0

    async def test_pin_skips_when_every_attempt_fails(self, router_ready):
        # [FUNCTION DOC] A single engine that fails ninety-nine times. The pin returns skip reason pin
        # violation (no crash), and no weight version query was ever made.
        """A fleet that never loads its weights must degrade to an attributable skip, not a crash."""
        log = []
        fleet = make_fleet(make_args(), [FlakyEngine(log, failures=99)])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"
        assert [e for e in log if e[0] == "get_weight_version"] == []

    async def test_pin_skips_when_a_weight_load_hangs_past_the_timeout(self, router_ready, monkeypatch):
        # [FUNCTION DOC] Shrinks the weight load timeout constant to 0.01 seconds and pins over one
        # engine whose load never completes. The pin returns skip reason pin violation, and the log
        # shows two load attempts (the initial one and one retry).
        """A wedged engine must not park the controller event loop forever, so the load is deadlined."""
        monkeypatch.setattr(eval_fleet_mod, "EVAL_WEIGHT_LOAD_TIMEOUT_SECS", 0.01)
        log = []
        fleet = make_fleet(make_args(), [HangingEngine(log)])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"
        assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 2

    async def test_pin_skips_when_the_fleet_reads_no_client_at_all(self, router_ready):
        # [FUNCTION DOC] A fleet with no engines must return skip reason pin violation instead of
        # passing vacuously.
        """An empty membership proves nothing about the served weights, so it must not pass the check vacuously."""
        fleet = make_fleet(make_args(), [])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"


class TestRouterProbe:
    # [CLASS DOC] Checks the router readiness probe sent during a pin.
    async def test_pin_probes_the_router_address_of_the_server_it_watches(self, router_probes):
        # [FUNCTION DOC] Sets the fake server's router to 10.9.9.9 port 39999, pins, and asserts success
        # with no skip reason, exactly one probe to the generate endpoint of that address, and a probe
        # payload requesting a single new token.
        """The fleet owns no address of its own, so the probe must use the one its server reports."""
        server = FakeEvalServer([FakeEngine([])])
        server.router_ip, server.router_port = "10.9.9.9", 39999
        fleet = InferenceControllerEvalFleet(make_args(), srv=server)

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)
        assert [probe.url for probe in router_probes] == ["http://10.9.9.9:39999/generate"]
        assert router_probes[0].json_payload["sampling_params"]["max_new_tokens"] == 1


class FakeInferenceController:
    # [CLASS DOC] HELPER: Fake inference controller handle. Answers pin requests from a scripted list in
    # call order, where an entry may be a pin result to return or an exception to raise, and records
    # each request's arguments.
    def __init__(self, pins: list[EvalFleetPin]):
        # [FUNCTION DOC] HELPER: Starts with an empty call list and stores the scripted answers.
        self.calls: list[dict] = []
        self._pins = pins

    async def pin_eval_fleet(self, *, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        # [FUNCTION DOC] HELPER: Records the checkpoint directory and weight version, takes the scripted
        # answer at the index of this call, raises it if it is an exception, otherwise returns it. Runs
        # out of answers with an index error if called more often than scripted.
        self.calls.append(dict(checkpoint_dir=checkpoint_dir, weight_version=weight_version))
        pin = self._pins[len(self.calls) - 1]
        if isinstance(pin, Exception):
            raise pin
        return pin


class FakeControllerProvider:
    # [CLASS DOC] HELPER: Fake provider of inference controller handles. Counts lookups and answers
    # from a scripted list by lookup number (the last entry repeats once the list is exhausted); an
    # entry that is an exception is raised instead of returned.
    def __init__(self, controllers):
        # [FUNCTION DOC] HELPER: Stores the scripted handles and starts the lookup counter at zero.
        self._controllers = controllers
        self.lookups = 0

    def get_handle(self, worker_name: str):
        # [FUNCTION DOC] HELPER: Increments the lookup counter, picks the entry at the lookup number
        # (clamped to the last entry), raises it if it is an exception, otherwise returns it. The
        # worker name is ignored.
        self.lookups += 1
        if isinstance(handle := self._controllers[min(self.lookups, len(self._controllers)) - 1], Exception):
            raise handle
        return handle


@pytest.fixture
def fleet_states(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Replaces the generate state constructor in the fleet module with
    # a function that appends its args to a list and returns a string naming the construction ordinal
    # (fake-state-1, fake-state-2, ...). Returns the list of args seen.
    built = []
    monkeypatch.setattr(eval_fleet_mod, "GenerateState", lambda args: built.append(args) or f"fake-state-{len(built)}")
    return built


def make_session(controller, *, info=None):
    # [FUNCTION DOC] HELPER: Builds an executor-side fleet over a provider that always hands out the
    # given controller.
    return make_session_over(FakeControllerProvider([controller]), info=info)


def make_session_over(provider, *, info=None):
    # [FUNCTION DOC] HELPER: Builds an executor-side fleet from test args, the given provider, and the
    # given fleet info (default: router 10.0.0.2 port 31000, two gpus, one gpu per engine).
    return RolloutExecutorEvalFleet(
        make_args(),
        info=info or EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1),
        inference_controller_provider=provider,
    )


class TestRolloutExecutorEvalFleet:
    # [CLASS DOC] Checks the executor-side eval fleet: it builds one generate state aimed at the fleet
    # router, pins through the inference controller handle (resolved once and reused), and maps
    # controller outcomes to returned state, skip errors or propagated errors.
    def test_builds_its_state_against_the_fleet_router(self, fleet_states):
        # [FUNCTION DOC] Creating a session builds exactly one generate state whose args carry the
        # fleet router ip and port and the fleet's gpu count and gpus per engine (2 and 1) instead of
        # the rollout values.
        """The executor generates against the eval router and the fleet's gpu sizing, not the rollout ones."""
        make_session(FakeInferenceController([]))

        (state_args,) = fleet_states
        assert (state_args.sglang_router_ip, state_args.sglang_router_port) == ("10.0.0.2", 31000)
        assert (state_args.rollout_num_gpus, state_args.rollout_num_gpus_per_engine) == (2, 1)

    async def test_pins_over_rpc_and_returns_the_cached_state(self, fleet_states):
        # [FUNCTION DOC] Two successful pins (snapshots step 5 and step 6) must each send the matching
        # checkpoint directory and weight version to the controller, return the same first state both
        # times, and construct only one state in total.
        """Pinning is the controller's call; the state is built once and handed back per point."""
        controller = FakeInferenceController([EvalFleetPin(skip_reason=None), EvalFleetPin(skip_reason=None)])
        session = make_session(controller)

        first = await session.pin("/snap/step_5", "5")
        second = await session.pin("/snap/step_6", "6")

        assert controller.calls == [
            dict(checkpoint_dir="/snap/step_5", weight_version="5"),
            dict(checkpoint_dir="/snap/step_6", weight_version="6"),
        ]
        assert first == second == "fake-state-1"
        assert len(fleet_states) == 1

    async def test_a_remote_skip_stays_an_attributable_skip(self, fleet_states):
        # [FUNCTION DOC] A controller answering with a pin violation skip reason makes the session raise
        # the eval skip error carrying the same reason.
        """The reason the controller skipped for must survive the wire as EvalSkip."""
        session = make_session(FakeInferenceController([EvalFleetPin(skip_reason="pin_violation")]))

        with pytest.raises(EvalSkip) as exc:
            await session.pin("/snap/step_5", "5")

        assert exc.value.reason == "pin_violation"

    async def test_the_controller_is_resolved_once_and_the_handle_reused(self, fleet_states):
        # [FUNCTION DOC] A provider scripted with two different controllers is used for two pins. The
        # first controller serves both pins, the second is never called, and the provider is looked up
        # exactly once.
        """Resolving a fresh handle per point would hand every point a fresh boot-uuid pin, so it is resolved once."""
        first, second = (
            FakeInferenceController([EvalFleetPin(skip_reason=None), EvalFleetPin(skip_reason=None)]),
            FakeInferenceController([EvalFleetPin(skip_reason=None)]),
        )
        provider = FakeControllerProvider([first, second])
        session = make_session_over(provider)

        await session.pin("/snap/step_5", "5")
        await session.pin("/snap/step_6", "6")

        assert (provider.lookups, len(first.calls), len(second.calls)) == (1, 2, 0)

    async def test_a_controller_that_cannot_be_reached_skips_the_point(self, fleet_states):
        # [FUNCTION DOC] A controller raising a worker unreachable error makes the pin raise the eval
        # skip error with reason controller unreachable.
        """Losing the controller must skip one eval point, not raise into the driver's rollout loop."""
        session = make_session(FakeInferenceController([WorkerUnreachableError("controller is gone")]))

        with pytest.raises(EvalSkip) as exc:
            await session.pin("/snap/step_5", "5")

        assert exc.value.reason == "controller_unreachable"

    async def test_a_controller_that_answered_with_a_failure_is_not_a_skip(self, fleet_states):
        # [FUNCTION DOC] A controller raising an rpc worker call error (the method failed inside a
        # reachable controller) must propagate unchanged and not be turned into a skip.
        """A method that raised inside a reachable controller is our bug, and skipping every point would hide it."""
        session = make_session(FakeInferenceController([RpcWorkerCallError("assert self._eval_fleet is not None")]))

        with pytest.raises(RpcWorkerCallError):
            await session.pin("/snap/step_5", "5")

    async def test_a_timed_out_controller_skips_the_point(self, fleet_states):
        # [FUNCTION DOC] A controller raising a timeout error makes the pin raise the eval skip error
        # with reason controller unreachable.
        """A controller that never answers is unreachable in every way that matters to one eval point."""
        session = make_session(FakeInferenceController([TimeoutError("no answer")]))

        with pytest.raises(EvalSkip) as exc:
            await session.pin("/snap/step_5", "5")

        assert exc.value.reason == "controller_unreachable"

    async def test_the_pin_keeps_reporting_a_restarted_controller_across_points(self, fleet_states):
        # [FUNCTION DOC] A controller scripted to raise a server restarted error on both calls. Two pins
        # for different snapshots each raise the eval skip error, and the provider was looked up only
        # once while the controller received both calls (the kept handle keeps reporting the restart).
        """The kept handle keeps its boot-uuid pin, so a later point still sees the restart instead of a fresh pin."""
        controller = FakeInferenceController(
            [ServerRestartedError("boot uuid changed"), ServerRestartedError("boot uuid changed")]
        )
        provider = FakeControllerProvider([controller])
        session = make_session_over(provider)

        for checkpoint_dir, weight_version in (("/snap/step_5", "5"), ("/snap/step_6", "6")):
            with pytest.raises(EvalSkip):
                await session.pin(checkpoint_dir, weight_version)

        assert (provider.lookups, len(controller.calls)) == (1, 2)
