# [MODULE DOC]
# Tests for the event-log views of the ft fault-injection support code. An event log records
# observation events (per-poll cell states) and injection events (fault attempts). The views
# compute: per-cell state histories with consecutive repeats collapsed, injection counts (harmed
# only by default, or all successful draws), per-form success counts, and the set of cells that
# were injected but did not show a fresh serving observation after their last injection (with a
# staleness grace period during which serving readings are ignored). Logs are built either through
# the shared helpers or from explicit timestamped events.
from datetime import datetime, timedelta, timezone

from tests.e2e.ft.conftest_ft.fault_injection import state, views
from tests.fast.e2e.ft.fault_injection.utils import (
    PENDING,
    RUNNING_NOT_SERVING,
    SERVING,
    SUSPENDED,
    log_of,
    note_injected,
    staged,
)


def test_observed_states_record_only_transitions() -> None:
    # [FUNCTION DOC] Observes serving, serving, suspended, suspended, serving for one engine and
    # asserts the state history is serving, suspended, serving.
    """Polling runs for the life of the training run, so repeats must not accumulate."""
    log = log_of([SERVING, SERVING, SUSPENDED, SUSPENDED, SERVING])

    assert views.compute_states_of_cell_name(log.events) == {"rollout-engine-0": [SERVING, SUSPENDED, SERVING]}


def test_a_serve_after_the_last_injection_clears_the_cell() -> None:
    # [FUNCTION DOC] Serving, pending (after an injection), serving with zero grace; asserts no
    # offending cells.
    """This is the recovery the soak asserts: the injected engine ends up serving again."""
    log = log_of([SERVING, PENDING, SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_a_completely_missed_down_window_is_not_an_offence() -> None:
    # [FUNCTION DOC] An injection lands but the cell is only ever observed serving; asserts no offenders.
    """A replacement can finish between two polls, and the witness must not demand the down it never saw."""
    log = log_of([SERVING, SERVING, SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_a_replacement_that_never_reaches_the_router_is_an_offence() -> None:
    # [FUNCTION DOC] Serving, pending (after injection), running-not-serving; asserts the cell is
    # reported as an offender with its full state history as values.
    """Regression: a relaunched engine stuck at PendingWeights also reads Running, and must not pass."""
    log = log_of([SERVING, PENDING, RUNNING_NOT_SERVING], inject_before={1: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {
        "rollout-engine-0": [SERVING.value, PENDING.value, RUNNING_NOT_SERVING.value]
    }


def test_a_cell_that_was_never_injected_owes_no_serve() -> None:
    # [FUNCTION DOC] Pending then serving with no injections; asserts the injection count is zero
    # and there are no offenders.
    """Otherwise a run that injected nothing would still fail the witness on ordinary churn."""
    log = log_of([PENDING, SERVING])

    assert views.compute_num_injections(log.events, cell_type="rollout") == 0
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_a_serve_that_predates_the_last_injection_does_not_discharge_it() -> None:
    # [FUNCTION DOC] Two injections (before the second and fourth observations), ending pending;
    # asserts the cell is an offender with its whole history, since the serve between them is older
    # than the last injection.
    """Otherwise the last crash of a soak is paid for by the recovery of the crash before it."""
    log = log_of([SERVING, PENDING, SERVING, PENDING], inject_before={1: 1, 3: 1})

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {
        "rollout-engine-0": [SERVING.value, PENDING.value, SERVING.value, PENDING.value]
    }


def test_only_the_last_injection_of_a_cell_needs_a_serve_after_it() -> None:
    # [FUNCTION DOC] Two injections followed by a final serving observation; asserts two counted
    # injections and no offenders.
    """Injections are serialized by the quiescence gate, so one final fresh serve settles the whole cell."""
    log = log_of([SERVING, PENDING, SERVING, SERVING], inject_before={1: 1, 3: 1})

    assert views.compute_num_injections(log.events, cell_type="rollout") == 2
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


def test_offences_of_another_cell_kind_do_not_count() -> None:
    # [FUNCTION DOC] An actor cell is observed serving, injected, then observed pending. Asserts
    # the rollout-type view reports no offenders while the actor-type view reports the actor with
    # its serving and pending history.
    """A mixed soak injects both kinds, and the rollout view must only see rollout cells."""
    log = state.EventLog()
    log.observe([staged("actor-0", SERVING, cell_type="actor")])
    note_injected(log, "actor-0")
    log.observe([staged("actor-0", PENDING, cell_type="actor")])

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}
    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="actor", grace_seconds=0.0) == {
        "actor-0": [SERVING.value, PENDING.value]
    }


def test_a_siblings_serve_cannot_clear_the_injected_cells_debt() -> None:
    # [FUNCTION DOC] Two engines serve; engine 0 is injected and goes pending while engine 1 keeps
    # serving. Asserts only engine 0 is an offender.
    """Only the victim's own fresh Serving reading proves its recovery."""
    log = state.EventLog()
    log.observe([staged("rollout-engine-0", SERVING), staged("rollout-engine-1", SERVING)])
    note_injected(log, "rollout-engine-0")
    log.observe([staged("rollout-engine-0", PENDING), staged("rollout-engine-1", SERVING)])

    offenders = views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0)
    assert set(offenders) == {"rollout-engine-0"}


class TestStaleServingGrace:
    # [CLASS DOC] Tests for the default staleness grace (120 seconds after an injection, per the
    # tests) during which serving readings do not count as recovery. Events carry explicit timestamps.
    def test_a_serve_inside_the_stale_window_does_not_clear_the_cell(self) -> None:
        # [FUNCTION DOC] Serve, injection at base time, serve 30 seconds later (inside the window);
        # asserts the cell is an offender with only a serving state in its history.
        """The api server reports a just-killed cell Serving for ~95s, so an early serve proves nothing."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=30)),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value]
        }

    def test_a_serve_exactly_at_the_grace_bound_clears_the_cell(self) -> None:
        # [FUNCTION DOC] A serve exactly 120 seconds after the injection counts as fresh; asserts no offenders.
        """The witness reads >= 120s as fresh, so a regression to a strict comparison must show here."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=120)),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {}

    def test_a_serve_reported_by_a_cell_that_is_not_healthy_does_not_clear_it(self) -> None:
        # [FUNCTION DOC] A serving reading 130 seconds after injection but from a cell flagged not
        # alive; asserts the cell is still an offender.
        """A cell that died without being deregistered keeps reporting Serving, and heals nothing."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=130), alive=False),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {
            "rollout-engine-0": [SERVING.value]
        }

    def test_a_serve_past_the_stale_window_clears_the_cell(self) -> None:
        # [FUNCTION DOC] A healthy serving reading 130 seconds after injection; asserts no offenders.
        """A reading older than the staleness bound is fresh, and it is the one that proves recovery."""
        base = datetime(2026, 8, 24, tzinfo=timezone.utc)
        events = [
            _observation("rollout-engine-0", SERVING, at=base),
            _injection("rollout-engine-0", at=base),
            _observation("rollout-engine-0", SERVING, at=base + timedelta(seconds=130)),
        ]

        assert views.compute_cells_not_serving_after_injection(events, cell_type="rollout") == {}


def _observation(
    name: str, cell_state: state.ObservedCellState, *, at: datetime, alive: bool = True
) -> state.ObservationsEvent:
    # [FUNCTION DOC] HELPER: Builds an observation event at the given time with one rollout-type
    # cell info (state and alive flag) keyed by name.
    return state.ObservationsEvent(
        timestamp=at, cell_infos={name: state.CellInfo(cell_type="rollout", state=cell_state, alive=alive)}
    )


def _injection(name: str, *, at: datetime) -> state.InjectionEvent:
    # [FUNCTION DOC] HELPER: Builds a successful, harmful injection event at the given time for the
    # named cell using the inject-fault sigkill form name.
    return state.InjectionEvent(
        timestamp=at, cell_name=name, form_name="inject_fault:sigkill", succeeded=True, harmed=True
    )


class TestWhichInjectionsCount:
    # [CLASS DOC] Tests for which injection attempts count: harmed-only count by default, all
    # successful draws on request, failed attempts never, and per-form success counts.
    def test_a_form_that_left_its_cell_running_is_not_a_crash_anything_has_to_heal(self) -> None:
        # [FUNCTION DOC] One successful but non-harmful hot-restart injection; asserts the default
        # injection count is zero and the cell owes no recovery.
        """A hot restart replaces the orchestration script and harms no cell, so no cell owes a recovery."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=True, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout") == 0
        assert (
            views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}
        )

    def test_a_form_that_left_its_cell_running_is_still_a_draw_that_fired(self) -> None:
        # [FUNCTION DOC] Same non-harmful success; asserts the count is one when harmed-only is disabled.
        """A soak counting what it actually did to the run has to see it, and asks for it by name."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=True, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 1

    def test_a_form_that_crashed_its_cell_counts_as_both(self) -> None:
        # [FUNCTION DOC] One successful harmful injection; asserts count one in both modes.
        """The crash forms every floor assertion was written for must go on counting as they did."""
        log = _log_of_one_injection(form_name="crash_pod", succeeded=True, harmed=True)

        assert views.compute_num_injections(log.events, cell_type="rollout") == 1
        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 1

    def test_a_draw_that_never_landed_counts_as_neither(self) -> None:
        # [FUNCTION DOC] One failed injection attempt; asserts the all-draws count is zero.
        """An attempt the cluster refused did nothing to the run, whatever the form would have done."""
        log = _log_of_one_injection(form_name="hot_restart", succeeded=False, harmed=False)

        assert views.compute_num_injections(log.events, cell_type="rollout", harmed_only=False) == 0

    def test_the_successes_of_one_form_are_counted_apart_from_another(self) -> None:
        # [FUNCTION DOC] Logs hot restart success, crash pod success and hot restart failure on one
        # engine; asserts each form's successful count is exactly one.
        """A mixed soak draws several forms, and each one's own assertions count only its own draws."""
        log = state.EventLog()
        log.observe([staged("rollout-engine-0", SERVING)])
        log.note_injection_attempt(cell_name="rollout-engine-0", form_name="hot_restart", succeeded=True, harmed=False)
        log.note_injection_attempt(cell_name="rollout-engine-0", form_name="crash_pod", succeeded=True, harmed=True)
        log.note_injection_attempt(
            cell_name="rollout-engine-0", form_name="hot_restart", succeeded=False, harmed=False
        )

        assert views.compute_num_successful_injections_of_form(log.events, form_name="hot_restart") == 1
        assert views.compute_num_successful_injections_of_form(log.events, form_name="crash_pod") == 1


def _log_of_one_injection(*, form_name: str, succeeded: bool, harmed: bool) -> state.EventLog:
    # [FUNCTION DOC] HELPER: Builds a log with a serving observation of one rollout engine, one
    # injection attempt with the given form name, success and harm flags, then another serving observation.
    log = state.EventLog()
    log.observe([staged("rollout-engine-0", SERVING)])
    log.note_injection_attempt(cell_name="rollout-engine-0", form_name=form_name, succeeded=succeeded, harmed=harmed)
    log.observe([staged("rollout-engine-0", SERVING)])
    return log
