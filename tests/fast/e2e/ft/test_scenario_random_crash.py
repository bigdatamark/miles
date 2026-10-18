# [MODULE DOC]
# Tests for the healing assertions of the ft random-crash scenario and the injection schedule it
# builds. Fake fault injector handles carry an event log that tests fill with synthetic cell
# observations and injection attempts; event directories are filled with synthetic reconfigure
# events written through the real event logger. Covers: the healing witness for trainer-only,
# rollout-only and mixed soaks (reconfigure pairing by cell, too-little-proof floor, serve
# freshness with a stale grace window), the mean injection interval per cell type, and the checks
# that every drawn or enabled fault form succeeded at least once.
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import entrypoint, fault_forms, state, views
from tests.e2e.ft.conftest_ft.scenario_random_crash import _assert_drawn_fault_forms_worked, assert_healing

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend

_ROLLOUT_CELL_NAME = "rollout-engine-0"
_ACTOR_CELL_NAME = "actor-0"


def _injector(
    *, cell_types: tuple[str, ...], cell_fault_forms: fault_forms.CellFaultForms | None = None
) -> entrypoint.FaultInjectorHandle:
    # [FUNCTION DOC] HELPER: Builds an unstarted fault injector handle for a fake control url, seed
    # zero, a huge mean interval for each given cell type, and the given fault forms (default:
    # a single sigkill api-server form per cell type).
    return entrypoint.FaultInjectorHandle(
        base_url="http://control",
        seed=0,
        mean_interval_seconds_of_cell_type={cell_type: 1e9 for cell_type in cell_types},
        cell_fault_forms=cell_fault_forms if cell_fault_forms is not None else _sigkill_forms(cell_types),
    )


def _sigkill_forms(cell_types: tuple[str, ...]) -> fault_forms.CellFaultForms:
    # [FUNCTION DOC] HELPER: Returns a mapping giving each cell type a single inject-fault form
    # with the sigkill failure mode for a fake control url.
    return {
        cell_type: [fault_forms.InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL)]
        for cell_type in cell_types
    }


def _all_forms_of_ray_run() -> fault_forms.CellFaultForms:
    # [FUNCTION DOC] HELPER: Returns the full set of fault forms the factory builds for a default
    # Ray-backend config.
    config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)
    return fault_forms.create_cell_fault_forms(base_url="http://control", config=config)


def _actor_cell(name: str = _ACTOR_CELL_NAME) -> dict:
    # [FUNCTION DOC] HELPER: Builds a fake running, healthy actor cell object with the given name
    # (default actor-0) and an initial workers hash label.
    return {
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": "actor", "miles.io/workers-hash": "generation-0"},
        },
        "status": {"phase": "Running", "conditions": [{"type": "Healthy", "status": "True"}]},
    }


def _note_actor_injections(
    injector: entrypoint.FaultInjectorHandle, count: int, *, name: str = _ACTOR_CELL_NAME
) -> None:
    # [FUNCTION DOC] HELPER: Appends to the injector's event log, count times, an observation of
    # the named actor cell followed by a successful sigkill injection attempt against it.
    log = injector.event_log
    for _ in range(count):
        log.observe([_actor_cell(name)])
        log.note_injection_attempt(
            cell_name=name,
            form_name="inject_fault:sigkill",
            succeeded=True,
        )


def _note_form_attempts(
    injector: entrypoint.FaultInjectorHandle, *, form_name: str, outcomes: list[bool], name: str = _ACTOR_CELL_NAME
) -> None:
    # [FUNCTION DOC] HELPER: Observes the named actor cell once, then records one injection
    # attempt of the given form per entry of outcomes, with that entry as the success flag.
    injector.event_log.observe([_actor_cell(name)])
    for succeeded in outcomes:
        injector.event_log.note_injection_attempt(
            cell_name=name,
            form_name=form_name,
            succeeded=succeeded,
        )


def _note_rollout_injection(log: state.EventLog) -> None:
    # [FUNCTION DOC] HELPER: Records a successful sigkill injection attempt against the rollout
    # engine cell in the given log.
    log.note_injection_attempt(
        cell_name=_ROLLOUT_CELL_NAME,
        form_name="inject_fault:sigkill",
        succeeded=True,
    )


def _rollout_cell(cell_state: state.ObservedCellState) -> dict:
    # [FUNCTION DOC] HELPER: Builds a fake rollout engine cell object in the given observed state.
    # Pending state gives phase Pending with no conditions; any other state gives phase Running with
    # a true Healthy condition and a Serving condition true only for the serving state.
    phase = "Pending" if cell_state is state.ObservedCellState.PENDING else "Running"
    conditions = (
        []
        if phase == "Pending"
        else [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if cell_state is state.ObservedCellState.SERVING else "False"},
        ]
    )
    return {
        "metadata": {
            "name": _ROLLOUT_CELL_NAME,
            "labels": {"miles.io/cell-type": "rollout", "miles.io/workers-hash": "generation-0"},
        },
        "status": {"phase": phase, "conditions": conditions},
    }


def _write_shrink_only_events(event_dir: Path) -> None:
    # [FUNCTION DOC] HELPER: Writes, through a real event logger in the given directory, a single
    # cell reconfigure event at rollout 2, quorum 1, no source cell, no healed cells, only cell 0
    # alive afterwards (a shrink without any healing). Closes the logger. Side effect: files on disk.
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="main"))
    event_logger.log(
        CellReconfigureEvent,
        dict(rollout_id=2, quorum_id=1, src_cell_index=None, healed_cell_indices=[], alive_cell_indices_after=[0]),
        print_log=False,
    )
    event_logger.close()


def _write_healing_events(event_dir: Path, healed_cell_indices_per_event: list[list[int]]) -> None:
    # [FUNCTION DOC] HELPER: Writes one cell reconfigure event per entry of the given list into the
    # directory through a real event logger. Event i has rollout id i plus 2, quorum id i plus 1,
    # source cell 0, the entry as its healed cell indices, and cells 0 and 1 alive afterwards.
    # Closes the logger. Side effect: files on disk.
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="main"))
    for index, healed_cell_indices in enumerate(healed_cell_indices_per_event):
        event_logger.log(
            CellReconfigureEvent,
            dict(
                rollout_id=index + 2,
                quorum_id=index + 1,
                src_cell_index=0,
                healed_cell_indices=healed_cell_indices,
                alive_cell_indices_after=[0, 1],
            ),
            print_log=False,
        )
    event_logger.close()


class TestAssertHealing:
    # [CLASS DOC] Tests of the top-level healing assertion for trainer-only, rollout-only and mixed soaks.
    def test_trainer_soak_rejects_missing_reconfigure_witness(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Three actor injections but only a shrink reconfigure event on disk; asserts
        # the healing assertion raises an assertion error saying the healing witness failed.
        """A trainer-only soak whose accepted injections produced no healing event must fail."""
        _write_shrink_only_events(tmp_path / "events")
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_trainer_soak_ignores_rollout_injections_when_counting_its_own(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Mixed soak with three rollout injections and no actor injections;
        # asserts an assertion error saying the soak proved too little.
        """A mixed soak's engine crashes say nothing about trainer healing, so they must not be counted."""
        _write_shrink_only_events(tmp_path / "events")
        injector = _injector(cell_types=("actor", "rollout"))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        for _ in range(3):
            _note_rollout_injection(log)

        with pytest.raises(AssertionError, match="Soak proved too little"):
            assert_healing(("train", "rollout"), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_rollout_soak_rejects_a_cell_never_seen_serving_after_its_last_injection(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Rollout history: serving, inject, pending, serving, inject, pending (ends
        # down after its last injection). Asserts an assertion error saying the rollout recovery
        # witness failed.
        """A rollout-only soak that ends with its last victim still relaunching must fail."""
        injector = _injector(cell_types=("rollout",))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])

        with pytest.raises(AssertionError, match="Rollout recovery witness failed"):
            assert_healing(("rollout",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_rollout_soak_accepts_a_fresh_serve_after_the_last_injection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Sets the views module's stale-status grace to zero seconds. History:
        # serving, two injections, pending, serving. Asserts the healing assertion passes without raising.
        """The witness must stay invisible on the path a healthy soak actually takes."""
        monkeypatch.setattr(views, "STALE_STATUS_GRACE_SECONDS", 0.0)
        injector = _injector(cell_types=("rollout",))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        for _ in range(2):
            _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.PENDING)])
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])

        assert_healing(("rollout",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_rollout_soak_rejects_a_serve_still_inside_the_stale_window(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] History: serving, two injections, serving, with the default stale grace
        # (readings land immediately after the kill). Asserts an assertion error saying the rollout
        # recovery witness failed.
        """A serve observed right after the kill can be the dead cell's stale reading, and proves nothing."""
        injector = _injector(cell_types=("rollout",))
        log = injector.event_log
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])
        for _ in range(2):
            _note_rollout_injection(log)
        log.observe([_rollout_cell(state.ObservedCellState.SERVING)])

        with pytest.raises(AssertionError, match="Rollout recovery witness failed"):
            assert_healing(("rollout",), injector=injector, event_dir=tmp_path / "events", context="soak")


def _mean_intervals(*ft_components: str) -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Computes the mean injection interval per cell type for the given ft
    # components with a trainer interval of 120 seconds and a rollout interval of 240 seconds.
    return fault_forms.compute_mean_interval_seconds_of_cell_type(
        tuple(ft_components), trainer_crash_interval_seconds=120.0, rollout_crash_interval_seconds=240.0
    )


def test_a_trainer_only_soak_schedules_actor_injections_only() -> None:
    # [FUNCTION DOC] Asserts train-only components give a schedule with only the actor type at 120 seconds.
    """It must not crash engines that its assertions say nothing about."""
    assert _mean_intervals("train") == {"actor": 120.0}


def test_a_rollout_only_soak_schedules_rollout_injections_only() -> None:
    # [FUNCTION DOC] Asserts rollout-only components give a schedule with only the rollout type at 240 seconds.
    """Crashing trainer cells here would exercise a component this mode did not enable ft on."""
    assert _mean_intervals("rollout") == {"rollout": 240.0}


def test_a_mixed_soak_keeps_each_kind_on_the_cadence_it_would_have_alone() -> None:
    # [FUNCTION DOC] Asserts both components give actor at 120 seconds and rollout at 240 seconds.
    """Adding rollout to a soak must not dilute the trainer crash rate it was calibrated at."""
    assert _mean_intervals("train", "rollout") == {"actor": 120.0, "rollout": 240.0}


def test_a_kind_the_mode_does_not_enable_ft_on_gets_no_schedule_at_all() -> None:
    # [FUNCTION DOC] Asserts the rollout type is absent from the schedule of a train-only mode.
    """An entry in the map is what makes the loop consider a kind, so a stray one crashes an unwatched component."""
    assert "rollout" not in _mean_intervals("train")


class TestAssertEveryDrawnFaultFormWorked:
    # [CLASS DOC] Tests of the check that every fault form that was drawn succeeded at least once.
    def test_a_form_that_never_worked_fails_the_soak(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Three successful kills plus four failed pod-deletion attempts; asserts an
        # assertion error naming the pod-deletion form.
        # KNOWN ISSUE: the temp path argument is unused.
        """Pod deletion can be refused for the whole run while the kills alone clear the injection floor."""
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)
        _note_form_attempts(injector, form_name=fault_forms.DELETE_POD_FORM_NAME, outcomes=[False] * 4)

        with pytest.raises(AssertionError, match=fault_forms.DELETE_POD_FORM_NAME):
            _assert_drawn_fault_forms_worked(injector)

    def test_a_form_that_worked_at_least_once_is_accepted(self) -> None:
        # [FUNCTION DOC] Pod-deletion attempts fail three times then succeed once; asserts the
        # check passes without raising.
        """A single refusal is a cluster hiccup, not proof the fault form is wired up wrong."""
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)
        _note_form_attempts(injector, form_name=fault_forms.DELETE_POD_FORM_NAME, outcomes=[False, False, False, True])

        _assert_drawn_fault_forms_worked(injector)


class TestAssertEveryEnabledFaultFormWorked:
    # [CLASS DOC] Tests that the healing assertion requires every enabled form of the enabled
    # components to have injected successfully at least once.
    def test_a_form_the_soak_never_drew_fails_it(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two healed events and two sigkill injections with all Ray forms available;
        # asserts an assertion error saying a form never injected successfully.
        """Regression: a soak that cleared the injection floor with one form used to pass without trying the rest."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_types=("actor",), cell_fault_forms=_all_forms_of_ray_run())
        _note_actor_injections(injector, 2)

        with pytest.raises(AssertionError, match="never injected successfully"):
            assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_a_soak_that_landed_every_enabled_form_passes(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Three healing events and one successful attempt of every failure-mode
        # form; asserts the healing assertion passes.
        """The happy path has to stay reachable, or the refusal above proves nothing."""
        _write_healing_events(tmp_path / "events", [[0], [0], [0]])
        injector = _injector(cell_types=("actor",), cell_fault_forms=_all_forms_of_ray_run())
        for failure_mode in fault_forms.FAILURE_MODES:
            _note_form_attempts(injector, form_name=f"inject_fault:{failure_mode.value}", outcomes=[True])

        assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_forms_of_a_component_the_mode_did_not_enable_are_not_required(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Injector knows actor and rollout forms but only train is enabled; every
        # failure-mode form of the actor succeeded; asserts the assertion passes.
        """A trainer-only soak must not be failed for never crashing an engine it was told to leave alone."""
        _write_healing_events(tmp_path / "events", [[0], [0], [0]])
        injector = _injector(cell_types=("actor", "rollout"), cell_fault_forms=_all_forms_of_ray_run())
        for failure_mode in fault_forms.FAILURE_MODES:
            _note_form_attempts(injector, form_name=f"inject_fault:{failure_mode.value}", outcomes=[True])

        assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")


class TestTrainerHealingPairing:
    # [CLASS DOC] Tests that trainer injections are paired with healings of the same cell index.
    def test_a_final_injection_that_never_healed_fails_even_though_the_floor_is_cleared(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Three injections of actor-0 but only two healing events; asserts an
        # assertion error saying the trainer recovery witness failed.
        """Regression: 3 crashes with 2 heals used to pass, leaving the run permanently degraded."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 3)

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_two_cells_healed_by_one_reconfigure_event_count_as_two_healings(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] One reconfigure event healing cells 0 and 1, with one injection each of
        # actor-0 and actor-1; asserts the assertion passes.
        """One reconfigure can readmit several cells, so counting events would under-count the healing."""
        _write_healing_events(tmp_path / "events", [[0, 1]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_healing_a_cell_that_was_never_injected_does_not_pay_another_cells_debt(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two healing events of cell 0 but two injections of actor-1; asserts an
        # assertion error saying the trainer recovery witness failed.
        """Counting healings without pairing them by cell index would call this a healthy soak."""
        _write_healing_events(tmp_path / "events", [[0], [0]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 2, name="actor-1")

        with pytest.raises(AssertionError, match="Trainer recovery witness failed"):
            assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")

    def test_every_injection_paired_with_a_healing_of_the_same_cell_passes(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Healing events for cell 0 then cell 1, with one injection of each; asserts
        # the assertion passes.
        """The assertion must stay invisible on the path a healthy soak actually takes."""
        _write_healing_events(tmp_path / "events", [[0], [1]])
        injector = _injector(cell_types=("actor",))
        _note_actor_injections(injector, 1, name="actor-0")
        _note_actor_injections(injector, 1, name="actor-1")

        assert_healing(("train",), injector=injector, event_dir=tmp_path / "events", context="soak")
