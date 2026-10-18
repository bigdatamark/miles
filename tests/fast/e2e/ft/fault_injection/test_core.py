# [MODULE DOC]
# Tests for the fault-injection loop in the ft fault-injection core module. The loop polls a fake
# control-plane api server (HTTP get/post patched out) for the list of cells, decides when each cell
# type is quiescent (all replicas serving for a required streak of polls), and injects faults into
# a chosen cell through its fault forms. Tests drive the loop with scripted listings and a stop
# event, and assert which cells were targeted, how injections were spaced, and how errors, stops,
# disabled injection and per-kind replica counts are handled.
import random
import threading
from collections.abc import Callable
from unittest.mock import MagicMock

from tests.e2e.ft.conftest_ft.fault_injection import core, fault_forms, state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    StubFaultForm,
    api_server_fault_forms,
    cell,
    fixed_fault_forms,
    intervals,
    mock_response,
    patched_requests,
    typed_cell,
)


def _run_injection_loop(
    *,
    fake_get,
    fake_post=None,
    cell_types: tuple[str, ...] = ("actor", "rollout"),
    quiescent_polls_required: int = 1,
    event_log: state.EventLog | None = None,
    cell_fault_forms: fault_forms.CellFaultForms | None = None,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    stop_event: threading.Event,
) -> None:
    # [FUNCTION DOC] HELPER: Runs the fault injection loop synchronously against a patched requests
    # module. The get side effect is fake_get and, if given, the post side effect is fake_post. The
    # loop uses a fixed fake base url, seed 0, a near-zero mean interval for each given cell type, a
    # fresh event log unless one is given, api-server fault forms unless forms are given, and a
    # microsecond poll interval. The loop returns once the stop event is set by a fake.
    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        if fake_post is not None:
            mock_requests.post.side_effect = fake_post
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(cell_types, 1e-9),
            stop_event=stop_event,
            event_log=event_log or state.EventLog(),
            cell_fault_forms=cell_fault_forms or api_server_fault_forms(),
            get_virtual_cells=get_virtual_cells,
            injection_enabled=injection_enabled,
            poll_interval_seconds=1e-6,
            quiescent_polls_required=quiescent_polls_required,
        )


def test_no_injection_before_the_quiescence_streak_is_long_enough() -> None:
    # [FUNCTION DOC] Two healthy actor cells are listed on every poll; the stop event is set at the
    # fourth poll while ten consecutive quiescent polls are required. Asserts no fault post was made.
    """The first reads after a disturbance can all be stale, so a short all-serving streak buys no kill."""
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Fake list call. Counts polls, sets the stop event from the fourth
        # poll on, and returns two healthy actor cells.
        polls["n"] += 1
        if polls["n"] >= 4:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Fake fault post. Records the target cell name parsed from the url
        # (the path segment after the cells part) and returns an empty ok response.
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    _run_injection_loop(fake_get=fake_get, fake_post=fake_post, quiescent_polls_required=10, stop_event=stop_event)

    assert injected == []


def test_successive_injections_are_spaced_by_the_quiescence_gate() -> None:
    # [FUNCTION DOC] Two healthy actor cells, required streak of three polls, stop at the fourteenth
    # poll. Records the poll number of each injection; asserts at least two injections and that the
    # gap between consecutive injections is at least the required streak.
    """An injection resets the streak, so the next kill of that kind waits out the gate again."""
    injection_polls: list[int] = []
    stop_event = threading.Event()
    required = 3
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops from the fourteenth, returns two healthy actors.
        polls["n"] += 1
        if polls["n"] >= 14:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Records the current poll count as the injection time; returns ok.
        injection_polls.append(polls["n"])
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=required,
        stop_event=stop_event,
    )

    assert len(injection_polls) >= 2, injection_polls
    gaps = [after - before for before, after in zip(injection_polls, injection_polls[1:], strict=False)]
    assert all(gap >= required for gap in gaps), injection_polls


def test_virtual_cells_use_the_regular_targeted_injection_path() -> None:
    # [FUNCTION DOC] The api server lists no cells; two synthetic virtual cells come from the virtual
    # cell provider, with a stub fault form that records the target name and stops the loop. Asserts
    # exactly one injection happened and it targeted one of the two virtual cells.
    """Synthetic replicas satisfy the ordinary scheduler without a real FT cell."""
    injected: list[str] = []
    stop_event = threading.Event()
    virtual_cells = [
        typed_cell("virtual-0", "virtual"),
        typed_cell("virtual-1", "virtual"),
    ]

    def inject(target: dict, _rng: random.Random) -> None:
        # [FUNCTION DOC] HELPER: Stub injection: records the target's metadata name and stops the loop.
        injected.append(target["metadata"]["name"])
        stop_event.set()

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Returns an empty cell list.
        return mock_response({"items": []})

    _run_injection_loop(
        fake_get=fake_get,
        cell_types=("virtual",),
        cell_fault_forms={"virtual": [StubFaultForm("virtual-fault", inject)]},
        get_virtual_cells=lambda: virtual_cells,
        stop_event=stop_event,
    )

    assert len(injected) == 1
    assert injected[0] in {"virtual-0", "virtual-1"}


def test_disabled_injection_still_observes_cells_without_injecting() -> None:
    # [FUNCTION DOC] Runs the loop with the injection-enabled callback returning false, two healthy
    # actors and stop at the third poll. Asserts no post was made and the event log still recorded
    # events (cells were observed).
    """A closing scenario keeps recovery evidence while admitting no new fault."""
    injected: list[str] = []
    event_log = state.EventLog()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops from the third, returns two healthy actors.
        polls["n"] += 1
        if polls["n"] >= 3:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Records the posted url; returns ok.
        injected.append(url)
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        event_log=event_log,
        injection_enabled=lambda: False,
        stop_event=stop_event,
    )

    assert injected == []
    assert event_log.events


def test_a_kind_with_a_dead_replica_is_not_quiescent() -> None:
    # [FUNCTION DOC] After each injection the fake marks the killed actor unhealthy for three
    # polls. Required streak is two, stop after two injections or 100 polls. Asserts at least two
    # injections and that the second came at least three plus two polls after the first (outage
    # length plus the quiescence streak).
    """A kill must be followed by an observed full recovery streak before that kind is due again."""
    injection_polls: list[int] = []
    stop_event = threading.Event()
    down = {"name": None, "polls_left": 0}
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops once two injections occurred or 100 polls passed.
        # Lists two actors, the one named in the down state being unhealthy while polls remain, and
        # decrements the remaining down polls.
        polls["n"] += 1
        if len(injection_polls) >= 2 or polls["n"] >= 100:
            stop_event.set()
        items = [cell(n, healthy=not (down["name"] == n and down["polls_left"] > 0)) for n in ("actor-0", "actor-1")]
        if down["polls_left"] > 0:
            down["polls_left"] -= 1
        return mock_response({"items": items})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Records the poll number, then marks the targeted cell (parsed from
        # the url) as down for three polls; returns ok.
        injection_polls.append(polls["n"])
        down["name"], down["polls_left"] = url.rsplit("/cells/", 1)[1].split("/")[0], 3
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=2,
        stop_event=stop_event,
    )

    assert len(injection_polls) >= 2, injection_polls
    assert injection_polls[1] - injection_polls[0] >= 3 + 2, injection_polls


def test_a_vanished_replica_blocks_its_kind_even_when_the_survivors_serve() -> None:
    # [FUNCTION DOC] After the first injection the killed cell vanishes from the listing while the
    # other two actors stay healthy; stop at poll twenty. Asserts only one injection ever happened.
    """A killed pod can disappear from the listing entirely, which must read as still recovering."""
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}
    all_names = ("actor-0", "actor-1", "actor-2")

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops from the twentieth. Lists all three actors, or
        # only those not yet injected once any injection happened; all healthy.
        polls["n"] += 1
        if polls["n"] >= 20:
            stop_event.set()
        names = [n for n in all_names if n not in injected] if injected else list(all_names)
        return mock_response({"items": [cell(n, healthy=True) for n in names]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Records the targeted cell name from the url; returns ok.
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=2,
        stop_event=stop_event,
    )

    assert len(injected) == 1, injected


def _run_typed_injection_loop(cells: list[dict], *, cell_types: tuple[str, ...], num_polls: int = 8) -> list[str]:
    # [FUNCTION DOC] HELPER: Runs the injection loop with a constant cell listing for the given
    # cell types, stopping at poll number num_polls (default 8). Returns the ordered list of
    # targeted cell names parsed from the post urls.
    injected: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, sets stop at num_polls, returns the fixed cells.
        polls["n"] += 1
        if polls["n"] >= num_polls:
            stop_event.set()
        return mock_response({"items": cells})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Records the targeted cell name; returns ok.
        injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
        return mock_response({})

    _run_injection_loop(fake_get=fake_get, fake_post=fake_post, cell_types=cell_types, stop_event=stop_event)

    return injected


def test_a_stop_that_arrives_while_listing_buys_no_further_injection() -> None:
    # [FUNCTION DOC] The list call itself sets the stop event while returning two actors. Asserts
    # that no fault was posted.
    """A fault injected on the way out is one nothing is left polling to see recover."""
    injected: list[str] = []
    stop_event = threading.Event()

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Sets the stop event and returns two typed actor cells.
        stop_event.set()
        return mock_response({"items": [typed_cell("actor-0", "actor"), typed_cell("actor-1", "actor")]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Records the url; returns ok.
        injected.append(url)
        return mock_response({})

    _run_injection_loop(fake_get=fake_get, fake_post=fake_post, stop_event=stop_event)

    assert injected == []


def test_injection_can_be_restricted_to_one_kind_of_cell() -> None:
    # [FUNCTION DOC] Lists two actors and two rollout engines but targets only the rollout type.
    # Asserts injections occurred and every target name starts with the rollout prefix.
    """Rollout and trainer cells share one api server, so a run targets one kind at a time."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
            typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_types=("rollout",),
    )

    assert injected
    assert all(name.startswith("rollout-") for name in injected), injected


def test_the_live_replica_count_only_considers_the_targeted_kind() -> None:
    # [FUNCTION DOC] Lists two actors and a single rollout engine, targeting rollout only. Asserts
    # no injection (the last rollout replica must be kept alive).
    """A single rollout cell must not be killed just because trainer cells are also alive."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_types=("rollout",),
    )

    assert injected == []


def test_a_mixed_run_sees_every_targeted_kind() -> None:
    # [FUNCTION DOC] Two actors and two rollout engines with both kinds targeted. Asserts at least
    # one injection happened.
    """A mixed-ft soak schedules both kinds, and must be able to crash either one."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
            typed_cell("rollout-engine-1", "rollout"),
        ],
        cell_types=("actor", "rollout"),
    )

    assert injected


def test_a_mixed_run_still_keeps_one_replica_of_each_kind() -> None:
    # [FUNCTION DOC] Two actors and one rollout engine with both kinds targeted. Asserts every
    # injection hit an actor (the single rollout engine is never killed).
    """Counting kinds together would let the trainer cells license killing the last engine."""
    injected = _run_typed_injection_loop(
        [
            typed_cell("actor-0", "actor"),
            typed_cell("actor-1", "actor"),
            typed_cell("rollout-engine-0", "rollout"),
        ],
        cell_types=("actor", "rollout"),
    )

    assert all(name.startswith("actor-") for name in injected), injected


class TestFaultInjectionLoopErrorHandling:
    # [CLASS DOC] Tests that transient failures of the list call or the fault post do not stop the
    # loop and are not counted as injections.
    def test_list_cells_failure_is_retried_and_does_not_stop_the_loop(self) -> None:
        # [FUNCTION DOC] The first list call raises a runtime error; later calls list two actors and
        # stop at poll six. Asserts some injection occurred and that the event log's injection
        # count for actors equals the number of posts made.
        """A transient api-server outage must cost one poll, not the rest of the soak."""
        cells = [typed_cell("actor-0", "actor"), typed_cell("actor-1", "actor")]
        log = state.EventLog()
        injected: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            # [FUNCTION DOC] HELPER: Counts polls; raises on the first, stops from the sixth,
            # otherwise returns the fixed cells.
            polls["n"] += 1
            if polls["n"] == 1:
                raise RuntimeError("api server unreachable")
            if polls["n"] >= 6:
                stop_event.set()
            return mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            # [FUNCTION DOC] HELPER: Records the targeted cell name; returns ok.
            injected.append(url.rsplit("/cells/", 1)[1].split("/")[0])
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("actor",),
            event_log=log,
            stop_event=stop_event,
        )

        assert injected, injected
        assert views.compute_num_injections(log.events, cell_type="actor") == len(injected)

    def test_failed_fault_post_is_not_counted_and_is_retried(self) -> None:
        # [FUNCTION DOC] The first fault post raises a runtime error; the second succeeds and stops
        # the loop. Asserts exactly two attempts and that the event log counts exactly one rollout
        # injection (the failed attempt is not counted).
        """A rejected inject-fault call must leave the soak free to try again, and must not inflate the tally."""
        cells = [typed_cell("rollout-engine-0", "rollout"), typed_cell("rollout-engine-1", "rollout")]
        log = state.EventLog()
        attempts: list[str] = []
        stop_event = threading.Event()
        polls = {"n": 0}

        def fake_get(url: str, timeout: float) -> MagicMock:
            # [FUNCTION DOC] HELPER: Counts polls, stops from the fifth, returns the fixed cells.
            polls["n"] += 1
            if polls["n"] >= 5:
                stop_event.set()
            return mock_response({"items": cells})

        def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
            # [FUNCTION DOC] HELPER: Records the attempt. Raises on the first; on later attempts
            # sets the stop event and returns ok.
            attempts.append(url.rsplit("/cells/", 1)[1].split("/")[0])
            if len(attempts) == 1:
                raise RuntimeError("inject-fault refused")
            stop_event.set()
            return mock_response({})

        _run_injection_loop(
            fake_get=fake_get,
            fake_post=fake_post,
            cell_types=("rollout",),
            event_log=log,
            stop_event=stop_event,
        )

        assert len(attempts) == 2, attempts
        assert views.compute_num_injections(log.events, cell_type="rollout") == 1


def test_a_failed_injection_forfeits_the_quiescence_streak() -> None:
    # [FUNCTION DOC] The first fault post raises; the required streak is three polls and the stop
    # comes at poll fourteen. Asserts at least two attempts and that the second attempt came at
    # least the required streak of polls after the failed first one.
    """A lost response does not prove a lost kill, so the next attempt must wait out the gate again."""
    attempt_polls: list[int] = []
    stop_event = threading.Event()
    required = 3
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops from the fourteenth, returns two healthy actors.
        polls["n"] += 1
        if polls["n"] >= 14:
            stop_event.set()
        return mock_response({"items": [cell("actor-0", healthy=True), cell("actor-1", healthy=True)]})

    def fake_post(url: str, json: dict, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Records the poll number of each attempt; raises on the first.
        attempt_polls.append(polls["n"])
        if len(attempt_polls) == 1:
            raise RuntimeError("response lost after the kill may have landed")
        return mock_response({})

    _run_injection_loop(
        fake_get=fake_get,
        fake_post=fake_post,
        cell_types=("actor",),
        quiescent_polls_required=required,
        stop_event=stop_event,
    )

    assert len(attempt_polls) >= 2, attempt_polls
    assert attempt_polls[1] - attempt_polls[0] >= required, attempt_polls


class TestMixedInjectionSelection:
    # [CLASS DOC] Tests which kind the loop picks when several kinds are targeted.
    def test_mixed_run_injects_rollout_when_only_rollout_has_a_spare(self) -> None:
        # [FUNCTION DOC] One actor and two rollout engines with both kinds targeted. Asserts
        # injections occurred and all targeted rollout engines.
        """The mirror of the trainer case: mixed selection must not be hard-coded to actor cells."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("actor-0", "actor"),
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout"),
            ],
            cell_types=("actor", "rollout"),
        )

        assert injected
        assert all(name.startswith("rollout-engine-") for name in injected), injected


def test_the_loop_injects_through_the_forms_of_the_cell_it_picked() -> None:
    # [FUNCTION DOC] Runs the loop directly with three actors and a single stub form named like the
    # delete-pod form that records its draws, stopping at poll six. Asserts the form was drawn, only
    # that form was used, and that the api server post route was never called.
    """A pod deletion drawn by the loop must reach kubectl, not the api server's inject-fault route."""
    drawn: list[str] = []
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops from the sixth, returns three typed actors.
        polls["n"] += 1
        if polls["n"] >= 6:
            stop_event.set()
        return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
            stop_event=stop_event,
            event_log=state.EventLog(),
            cell_fault_forms=fixed_fault_forms(
                [
                    StubFaultForm(
                        fault_forms.DELETE_POD_FORM_NAME,
                        lambda cell, rng: drawn.append(fault_forms.DELETE_POD_FORM_NAME),
                    )
                ]
            ),
            poll_interval_seconds=1e-6,
            quiescent_polls_required=1,
        )

        assert drawn, drawn
        assert set(drawn) == {fault_forms.DELETE_POD_FORM_NAME}, drawn
        mock_requests.post.assert_not_called()


def test_the_loop_draws_a_form_that_has_never_worked_before_repeating_a_proven_one() -> None:
    # [FUNCTION DOC] Four actors with three stub forms a, b, c that record their draws; stops at
    # poll ten. Asserts the first three draws cover all three forms (untried forms are preferred).
    """Uniform sampling can leave the rarest fault untried for a whole soak, which is the one worth trying."""
    drawn: list[str] = []
    log = state.EventLog()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops from the tenth, returns four typed actors.
        polls["n"] += 1
        if polls["n"] >= 10:
            stop_event.set()
        return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(4)]})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
            stop_event=stop_event,
            event_log=log,
            cell_fault_forms=fixed_fault_forms(
                [StubFaultForm(name, lambda cell, rng, n=name: drawn.append(n)) for name in ("a", "b", "c")]
            ),
            poll_interval_seconds=1e-6,
            quiescent_polls_required=1,
        )

    assert set(drawn[:3]) == {"a", "b", "c"}, drawn


def test_a_form_that_always_refuses_keeps_being_drawn_so_the_soak_can_see_it() -> None:
    # [FUNCTION DOC] Three actors with one working stub form and one that always raises; stops at
    # poll eight. Asserts the views helper reports exactly the broken form for actors as drawn
    # without ever succeeding.
    """A form that rides on the ones that did work would end the run green while never having fired."""
    log = state.EventLog()
    stop_event = threading.Event()
    polls = {"n": 0}

    def fake_get(url: str, timeout: float) -> MagicMock:
        # [FUNCTION DOC] HELPER: Counts polls, stops from the eighth, returns three typed actors.
        polls["n"] += 1
        if polls["n"] >= 8:
            stop_event.set()
        return mock_response({"items": [typed_cell(f"actor-{i}", "actor") for i in range(3)]})

    with patched_requests() as mock_requests:
        mock_requests.get.side_effect = fake_get
        core.run_fault_injection_loop(
            base_url="http://control",
            seed=0,
            mean_interval_seconds_of_cell_type=intervals(("actor", "rollout"), 1e-12),
            stop_event=stop_event,
            event_log=log,
            cell_fault_forms=fixed_fault_forms(
                [StubFaultForm("works", _do_nothing), StubFaultForm("broken", _always_refuse)]
            ),
            poll_interval_seconds=1e-6,
            quiescent_polls_required=1,
        )

    assert views.compute_forms_drawn_without_success(log.events) == [("actor", "broken")]


def _always_refuse(cell: dict, rng: random.Random) -> None:
    # [FUNCTION DOC] HELPER: Stub fault action that always raises a runtime error.
    raise RuntimeError("this form never works")


def _do_nothing(cell: dict, rng: random.Random) -> None:
    # [FUNCTION DOC] HELPER: Stub fault action that does nothing and returns nothing.
    return None


class TestRolloutQuiescence:
    # [CLASS DOC] Tests that rollout engines count as quiescent only when serving (in the router),
    # while trainer cells are judged by liveness alone.
    def test_an_engine_that_is_not_in_the_router_blocks_its_kind(self) -> None:
        # [FUNCTION DOC] Two rollout engines, one marked not serving. Asserts no injection.
        """A relaunched engine reads Healthy long before it can answer, so its kind is still recovering."""
        injected = _run_typed_injection_loop(
            [
                typed_cell("rollout-engine-0", "rollout"),
                typed_cell("rollout-engine-1", "rollout", serving=False),
            ],
            cell_types=("rollout",),
        )

        assert injected == []

    def test_two_serving_engines_still_leave_one_of_them_injectable(self) -> None:
        # [FUNCTION DOC] Two serving rollout engines. Asserts an injection happens.
        """The quiescence rule must not block the case it was never meant to block."""
        injected = _run_typed_injection_loop(
            [typed_cell("rollout-engine-0", "rollout"), typed_cell("rollout-engine-1", "rollout")],
            cell_types=("rollout",),
        )

        assert injected

    def test_a_trainer_cell_is_judged_by_liveness_alone(self) -> None:
        # [FUNCTION DOC] Asserts the core module's private serve-check treats a typed actor cell as
        # able to serve (no serving condition is required for trainer cells).
        """Trainer cells carry no Serving condition, so requiring one would stop every trainer soak."""
        assert core._cell_can_serve(typed_cell("actor-0", "actor"))
