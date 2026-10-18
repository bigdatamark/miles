# [MODULE DOC]
# Tests for the mini fault-tolerance controller. Parts: (1) mapping an API-server cell object to a
# cell snapshot with a health status (healthy condition true gives healthy, false gives unhealthy,
# unknown does not trigger healing, no healthy condition gives not-applicable, except a Suspended
# phase which is unhealthy so it gets resumed, and a Pending phase with a false healthy condition
# is unhealthy); (2) the polling controller: it asks a get-cells callable for snapshots each poll
# and, for unhealthy cells, calls suspend then resume; it skips healthy and not-applicable cells;
# failed heals set per-cell exponential backoff (base 5 s doubling per failure, capped at 300 s,
# next attempt time stored per cell), a cell inside its backoff window is skipped, success resets
# the backoff, and backoff entries are dropped when the cell leaves the unhealthy state; poll
# errors do not stop the loop and a stop request ends it; (3) the HTTP runner: parsing a cell list
# response, propagating HTTP errors, and sending suspend/resume patches with the right body to the
# right path; (4) argument defaults for the API server port and controller enablement; (5) starting
# the controller on a daemon thread. A fake clock and mocked async callables drive the controller.
from __future__ import annotations

import argparse
import asyncio
import json
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from miles.utils.arguments import (
    _DEFAULT_FT_API_SERVER_PORT,
    _resolve_api_server_port,
    _resolve_mini_ft_controller_enable,
)
from miles.utils.ft_utils import mini_ft_controller
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.ft_utils.mini_ft_controller import (
    CellHealthStatus,
    _CellSnapshot,
    _compute_cell_snapshot,
    _MiniFTController,
    _MiniFTControllerRunner,
)

# ------------------------ helpers ------------------------


def _make_cell(
    *,
    name: str = "cell-0",
    status: CellHealthStatus = CellHealthStatus.HEALTHY,
) -> _CellSnapshot:
    # [FUNCTION DOC] HELPER: Build a cell snapshot with the given name (default "cell-0") and health
    # status (default healthy).
    return _CellSnapshot(name=name, status=status)


HEALTHY = CellHealthStatus.HEALTHY
UNHEALTHY = CellHealthStatus.UNHEALTHY
NOT_APPLICABLE = CellHealthStatus.NOT_APPLICABLE


class _FakeClock:
    # [CLASS DOC] HELPER: Controllable clock: calling it returns the current time value, which tests
    # set directly or advance.
    def __init__(self, start: float = 0.0) -> None:
        # [FUNCTION DOC] HELPER: Set the current time to the start value (default zero).
        self.now = start

    def __call__(self) -> float:
        # [FUNCTION DOC] HELPER: Return the current time value.
        return self.now

    def advance(self, seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Add the given number of seconds to the current time.
        self.now += seconds


def _make_controller(
    *,
    get_cells: AsyncMock | None = None,
    suspend_cell: AsyncMock | None = None,
    resume_cell: AsyncMock | None = None,
    poll_interval: float = 0.01,
    resume_delay: float = 0.0,
    clock: _FakeClock | None = None,
) -> _MiniFTController:
    # [FUNCTION DOC] HELPER: Build the controller with the given async callables (defaults: a
    # get-cells returning an empty list, and plain async mocks for suspend and resume), poll
    # interval 0.01 s, resume delay zero and a fake clock starting at zero unless given.
    return _MiniFTController(
        get_cells=get_cells or AsyncMock(return_value=[]),
        suspend_cell=suspend_cell or AsyncMock(),
        resume_cell=resume_cell or AsyncMock(),
        poll_interval=poll_interval,
        resume_delay=resume_delay,
        clock=clock or _FakeClock(),
    )


def _run_controller_for_n_polls(
    controller: _MiniFTController,
    get_cells_mock: AsyncMock,
    responses: list[list[_CellSnapshot]],
) -> None:
    # [FUNCTION DOC] HELPER: Install a side effect on the get-cells mock that returns the scripted
    # responses in order (an empty list once they run out) and requests the controller to stop as
    # soon as the poll count reaches the number of responses, so the controller's run finishes
    # after exactly that many polls. Mutates the mock; returns nothing.
    """Configure get_cells to return `responses` then stop the controller."""
    poll_count = 0

    async def _side_effect() -> list[_CellSnapshot]:
        # [FUNCTION DOC] HELPER: Count the poll, pick the scripted response, and request a stop on
        # the final scripted poll.
        nonlocal poll_count
        result = responses[poll_count] if poll_count < len(responses) else []
        poll_count += 1
        if poll_count >= len(responses):
            controller.request_stop()
        return result

    get_cells_mock.side_effect = _side_effect


def _build_cell_json(
    *,
    name: str = "actor-0",
    healthy_status: str = "True",
    healthy_reason: str | None = None,
    workers_hash: str = "hash-0",
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build the JSON-style mapping of one cell as the API server returns it:
    # API version and kind, metadata with name and cell type/id labels, a spec with suspend false,
    # and a status with phase "Running", the workers hash, and two conditions: Allocated true and
    # Healthy with the given status string (default "True") and optional reason.
    return {
        "apiVersion": "miles.io/v1",
        "kind": "Cell",
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": "actor", "miles.io/cell-id": name},
        },
        "spec": {"suspend": False},
        "status": {
            "phase": "Running",
            "workers_hash": workers_hash,
            "conditions": [
                {"type": "Allocated", "status": "True"},
                {"type": "Healthy", "status": healthy_status, "reason": healthy_reason},
            ],
        },
    }


def _build_cell_list_json(cells: list[dict[str, Any]]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Wrap the given cell mappings in a cell-list mapping with API version
    # and kind and the cells under the items key.
    return {"apiVersion": "miles.io/v1", "kind": "CellList", "items": cells}


def _create_runner() -> _MiniFTControllerRunner:
    # [FUNCTION DOC] HELPER: Build a runner pointed at a local API server URL (port 8080) with a
    # 10 s poll interval, 5 s resume delay and automatic cell resume disabled.
    return _MiniFTControllerRunner(
        api_server_url="http://127.0.0.1:8080",
        poll_interval=10.0,
        resume_delay=5.0,
        cells_auto_resume=False,
    )


def _mock_response(*, status_code: int = 200, json_data: Any = None) -> httpx.Response:
    # [FUNCTION DOC] HELPER: Build a mock HTTP response with the given status code (default 200)
    # whose JSON method returns the given data and whose raise-for-status does nothing for codes
    # below 400 and raises an HTTP status error carrying the response for codes of 400 or more.
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=response,
        )
    return response


# ------------------------ snapshot tests ------------------------


def _make_cell_object(
    *, name: str = "actor-0", healthy: list[CellCondition] | None = None, phase: str = "Running"
) -> Cell:
    # [FUNCTION DOC] HELPER: Build a real cell model object with the given name (no labels), an
    # empty spec, the given phase (default "Running") and a conditions list that always starts with
    # an Allocated-true condition followed by the supplied extra (health) conditions, if any. The
    # workers hash is a fixed pseudo value.
    conditions: list[CellCondition] = [CellCondition.allocated(TriState.TRUE)]
    if healthy is not None:
        conditions.extend(healthy)
    return Cell(
        metadata=CellMetadata(name=name, labels={}),
        spec=CellSpec(),
        status=CellStatus(phase=phase, conditions=conditions, workers_hash="pseudo-hash-0"),
    )


class TestComputeCellSnapshot:
    # [CLASS DOC] Tests of the function that converts a cell object into a name-and-health snapshot.
    def test_healthy_true_maps_to_healthy(self):
        # [FUNCTION DOC] A cell with a Healthy-true condition yields a healthy snapshot.
        cell = _make_cell_object(healthy=[CellCondition.healthy(TriState.TRUE)])

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=HEALTHY)

    def test_healthy_false_maps_to_unhealthy(self):
        # [FUNCTION DOC] A cell with a Healthy-false condition (reason given) yields an unhealthy
        # snapshot.
        cell = _make_cell_object(healthy=[CellCondition.healthy(TriState.FALSE, reason="HealthCheckFailed")])

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=UNHEALTHY)

    def test_healthy_unknown_does_not_trigger_heal(self):
        # [FUNCTION DOC] A Healthy-unknown condition must not yield an unhealthy snapshot (only
        # asserts the status is not unhealthy).
        """Regression: UNKNOWN is a transient state (e.g. health checker paused
        during healing); controller must not treat it as unhealthy and reconcile."""
        cell = _make_cell_object(healthy=[CellCondition.healthy(TriState.UNKNOWN, reason="HealthCheckUnknown")])

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot.status != UNHEALTHY

    def test_missing_healthy_condition_maps_to_not_applicable(self):
        # [FUNCTION DOC] A running cell with no Healthy condition yields a not-applicable snapshot.
        cell = _make_cell_object(healthy=None)

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=NOT_APPLICABLE)

    def test_suspended_cell_maps_to_unhealthy_so_controller_resumes(self):
        # [FUNCTION DOC] A cell in phase "Suspended" with no Healthy condition yields an unhealthy
        # snapshot.
        """A Suspended cell was shrunk out by the train side (cell.stop()) and must be resumed;
        it carries no Healthy condition, so the controller must heal it by phase, not be misled
        into NOT_APPLICABLE (which would lose the cell forever)."""
        cell = _make_cell_object(healthy=None, phase="Suspended")

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=UNHEALTHY)

    def test_pending_cell_maps_to_not_applicable_not_healed(self):
        # [FUNCTION DOC] A cell in phase "Pending" with no Healthy condition yields a
        # not-applicable snapshot.
        """A Pending cell is mid-heal (being re-allocated); it also has no Healthy condition but
        must NOT be treated as needing heal, otherwise the controller fights the ongoing heal."""
        cell = _make_cell_object(healthy=None, phase="Pending")

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=NOT_APPLICABLE)

    def test_a_pending_cell_that_blew_its_startup_deadline_is_healed(self):
        # [FUNCTION DOC] A "Pending" cell with a Healthy-false condition (reason startup deadline
        # exceeded) yields an unhealthy snapshot.
        """A rollout cell stuck booting reports Pending plus Healthy=False; without healing it is
        stranded until the next hour-long readiness timeout."""
        cell = _make_cell_object(
            healthy=[CellCondition.healthy(TriState.FALSE, reason="StartupDeadlineExceeded")], phase="Pending"
        )

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=UNHEALTHY)

    def test_any_false_among_multiple_conditions_wins(self):
        # [FUNCTION DOC] A cell with both a Healthy-true and a Healthy-false condition yields an
        # unhealthy snapshot.
        """If any Healthy condition is FALSE, the cell is unhealthy regardless of
        other TRUE/UNKNOWN conditions — fail-loud over fail-quiet."""
        cell = _make_cell_object(
            healthy=[
                CellCondition.healthy(TriState.TRUE),
                CellCondition.healthy(TriState.FALSE),
            ]
        )

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot.status == UNHEALTHY


# ------------------------ controller tests ------------------------


class TestControllerHealing:
    # [CLASS DOC] Tests of which cells the polling controller heals by suspend then resume.
    @pytest.mark.asyncio
    async def test_heal_on_unhealthy_cell(self) -> None:
        # [FUNCTION DOC] One poll returning one unhealthy cell: suspend and resume are each called
        # exactly once with that cell's name. The controller run is bounded to 5 s.
        """Unhealthy cell triggers suspend → resume."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_called_once_with("cell-0")
        resume_cell.assert_called_once_with("cell-0")

    @pytest.mark.asyncio
    async def test_skip_healthy_cell(self) -> None:
        # [FUNCTION DOC] One poll returning a healthy cell: neither suspend nor resume is called.
        """Healthy cell does not trigger heal."""
        healthy_cell = _make_cell(name="cell-0", status=HEALTHY)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[healthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()

    @pytest.mark.asyncio
    async def test_heal_multiple_unhealthy_cells(self) -> None:
        # [FUNCTION DOC] One poll with two unhealthy cells: suspend and resume are each called twice,
        # once per cell name.
        """Multiple unhealthy cells are each healed."""
        cells = [
            _make_cell(name="cell-0", status=UNHEALTHY),
            _make_cell(name="cell-1", status=UNHEALTHY),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert suspend_cell.call_count == 2
        assert resume_cell.call_count == 2
        suspend_cell.assert_any_call("cell-0")
        suspend_cell.assert_any_call("cell-1")
        resume_cell.assert_any_call("cell-0")
        resume_cell.assert_any_call("cell-1")

    @pytest.mark.asyncio
    async def test_no_action_when_all_healthy(self) -> None:
        # [FUNCTION DOC] One poll with two healthy cells: neither suspend nor resume is called.
        """All healthy → no suspend/resume calls."""
        cells = [
            _make_cell(name="cell-0", status=HEALTHY),
            _make_cell(name="cell-1", status=HEALTHY),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()


class TestControllerBackoff:
    # [CLASS DOC] Tests of per-cell exponential backoff after failed heals, using a fake clock.
    @pytest.mark.asyncio
    async def test_backoff_on_heal_failure(self) -> None:
        # [FUNCTION DOC] Clock at 100; suspend raises a runtime error. After one poll the cell's
        # backoff record has one consecutive failure and next attempt time 110 (100 plus 5 times
        # two to the power one).
        """Suspend raises → consecutive_failures increments, next_attempt_at increases."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=100.0)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock(side_effect=RuntimeError("connection failed"))

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        backoff = controller._cell_backoffs["cell-0"]
        assert backoff.consecutive_failures == 1
        assert backoff.next_attempt_at == 110.0  # 100 + 5*2^1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        # [FUNCTION DOC] With suspend always failing, run one poll per iteration, moving the clock
        # to each previous next-attempt time. After each failure the next attempt time equals the
        # current clock plus the expected delay 10, 20, 40, 80, 160, 300, 300 (delay 5 times two to
        # the failure count, capped at 300).
        """Verify backoff delays: 5*2^1=10, 5*2^2=20, ..., capped at 300."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=0.0)
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        expected_delays = [10, 20, 40, 80, 160, 300, 300]
        for expected_delay in expected_delays:
            # Advance clock past backoff window so the heal attempt is made
            backoff = controller._cell_backoffs.get("cell-0")
            if backoff:
                clock.now = backoff.next_attempt_at

            _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
            await asyncio.wait_for(controller.run(), timeout=5.0)

            backoff = controller._cell_backoffs["cell-0"]
            assert backoff.next_attempt_at == clock.now + expected_delay

    @pytest.mark.asyncio
    async def test_skips_heal_when_within_backoff_window(self) -> None:
        # [FUNCTION DOC] Failing suspend, three runs: the first attempts a heal (one call); a second
        # poll without advancing the clock makes no new call; after moving the clock to the stored
        # next-attempt time a third poll attempts again (two calls total).
        """Cell is not healed again until clock passes next_attempt_at."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=0.0)
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        # Step 1: First poll → heal attempt fails, sets next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 1

        # Step 2: Poll again without advancing clock → should skip heal
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 1  # no new call

        # Step 3: Advance clock past backoff → should attempt heal again
        backoff = controller._cell_backoffs["cell-0"]
        clock.now = backoff.next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_heal_resets_backoff(self) -> None:
        # [FUNCTION DOC] Suspend fails on its first call and succeeds afterwards. After the failed
        # poll the backoff record shows one failure; after moving the clock to the next attempt
        # time and polling again, the same record shows zero failures and its next attempt time
        # equals the current clock.
        """Successful heal resets consecutive_failures to 0."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=0.0)

        call_count = 0

        async def failing_then_succeeding_suspend(name: str) -> None:
            # [FUNCTION DOC] HELPER: Count calls and raise a runtime error only on the first call;
            # the cell name is ignored.
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=AsyncMock(side_effect=failing_then_succeeding_suspend),
            resume_cell=AsyncMock(),
            clock=clock,
        )

        # Step 1: First attempt fails
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)

        backoff = controller._cell_backoffs["cell-0"]
        assert backoff.consecutive_failures == 1

        # Step 2: Advance clock past backoff, second attempt succeeds
        clock.now = backoff.next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert backoff.consecutive_failures == 0
        assert backoff.next_attempt_at == clock.now


class TestControllerNotApplicable:
    # [CLASS DOC] Tests of how not-applicable cells are skipped and how backoff records are cleaned.
    @pytest.mark.asyncio
    async def test_not_applicable_cell_skipped(self) -> None:
        # [FUNCTION DOC] One poll with a not-applicable cell: neither suspend nor resume is called.
        """NOT_APPLICABLE cell does not trigger heal."""
        na_cell = _make_cell(name="cell-0", status=NOT_APPLICABLE)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[na_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_statuses_only_heals_unhealthy(self) -> None:
        # [FUNCTION DOC] One poll with a healthy, an unhealthy and a not-applicable cell: suspend and
        # resume are called once each, only for the unhealthy cell.
        """HEALTHY and NOT_APPLICABLE skipped, only UNHEALTHY healed."""
        cells = [
            _make_cell(name="cell-0", status=HEALTHY),
            _make_cell(name="cell-1", status=UNHEALTHY),
            _make_cell(name="cell-2", status=NOT_APPLICABLE),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_called_once_with("cell-1")
        resume_cell.assert_called_once_with("cell-1")

    @pytest.mark.asyncio
    async def test_backoff_cleared_when_cell_leaves_unhealthy(self) -> None:
        # [FUNCTION DOC] A failed heal creates a backoff record for the cell; a following poll
        # where the cell is not-applicable removes the record.
        """Cell goes UNHEALTHY (heal fails) → NOT_APPLICABLE → backoff is cleaned up."""
        clock = _FakeClock(start=0.0)
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        na_cell = _make_cell(name="cell-0", status=NOT_APPLICABLE)

        get_cells = AsyncMock()
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        # Step 1: Unhealthy → heal fails → backoff entry created
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" in controller._cell_backoffs

        # Step 2: Cell becomes NOT_APPLICABLE → backoff entry cleaned
        _run_controller_for_n_polls(controller, get_cells, [[na_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" not in controller._cell_backoffs

    @pytest.mark.asyncio
    async def test_backoff_cleared_when_cell_becomes_healthy(self) -> None:
        # [FUNCTION DOC] A failed heal creates a backoff record; a following poll where the cell is
        # healthy removes the record.
        """Cell goes UNHEALTHY (heal fails) → HEALTHY → backoff is cleaned up."""
        clock = _FakeClock(start=0.0)
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        healthy_cell = _make_cell(name="cell-0", status=HEALTHY)

        get_cells = AsyncMock()
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        # Step 1: Unhealthy → heal fails → backoff entry created
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" in controller._cell_backoffs

        # Step 2: Cell becomes HEALTHY → backoff entry cleaned
        _run_controller_for_n_polls(controller, get_cells, [[healthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" not in controller._cell_backoffs


class TestControllerLifecycle:
    # [CLASS DOC] Tests of the controller loop's resilience and stop handling.
    @pytest.mark.asyncio
    async def test_poll_continues_after_get_cells_failure(self) -> None:
        # [FUNCTION DOC] The get-cells callable raises a runtime error twice, then on the third call
        # requests a stop and returns an empty list. Run completes within 5 s and exactly three
        # calls were made (errors do not end the loop).
        """get_cells raises → controller does not exit, continues polling."""
        call_count = 0

        async def failing_then_stopping(controller_ref: list[_MiniFTController]) -> list[_CellSnapshot]:
            # [FUNCTION DOC] HELPER: Count calls; raise a runtime error ("network error") for the
            # first two; on the third request the controller (held in a one-element list) to stop
            # and return an empty list.
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("network error")
            controller_ref[0].request_stop()
            return []

        controller = _make_controller()
        controller_ref = [controller]

        async def _side_effect() -> list[_CellSnapshot]:
            # [FUNCTION DOC] HELPER: Delegate to the failing-then-stopping function with the
            # controller reference.
            return await failing_then_stopping(controller_ref)

        controller._get_cells = AsyncMock(side_effect=_side_effect)

        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_request_stop_exits_loop(self) -> None:
        # [FUNCTION DOC] One poll returning no cells requests a stop; the run returns within 5 s.
        """call request_stop → run() returns."""
        get_cells = AsyncMock()
        controller = _make_controller(get_cells=get_cells)
        _run_controller_for_n_polls(controller, get_cells, [[]])

        await asyncio.wait_for(controller.run(), timeout=5.0)


# ------------------------ runner tests ------------------------


class TestRunnerGetCells:
    # [CLASS DOC] Tests of the runner's fetching and parsing of cells over HTTP (client mocked).
    @pytest.mark.asyncio
    async def test_parses_healthy_and_unhealthy(self) -> None:
        # [FUNCTION DOC] The mocked client returns a list of an actor with Healthy "True" and one
        # with "False"; the runner returns two snapshots in order: healthy then unhealthy, with the
        # cell names.
        """Parse CellList JSON into _CellSnapshot with correct healthy bool."""
        runner = _create_runner()

        cells_json = _build_cell_list_json(
            [
                _build_cell_json(name="actor-0", healthy_status="True"),
                _build_cell_json(name="actor-1", healthy_status="False"),
            ]
        )

        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

        result = await runner._get_cells()

        assert len(result) == 2
        assert result[0] == _CellSnapshot(name="actor-0", status=HEALTHY)
        assert result[1] == _CellSnapshot(name="actor-1", status=UNHEALTHY)

    @pytest.mark.asyncio
    async def test_missing_healthy_condition_treated_as_not_applicable(self) -> None:
        # [FUNCTION DOC] A cell whose only condition is Allocated yields one not-applicable snapshot.
        """Cell with no Healthy condition → status=NOT_APPLICABLE (not healed)."""
        runner = _create_runner()

        cell_json = _build_cell_json(name="actor-0")
        # Remove the Healthy condition, keep only Allocated
        cell_json["status"]["conditions"] = [{"type": "Allocated", "status": "True"}]

        cells_json = _build_cell_list_json([cell_json])
        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

        result = await runner._get_cells()

        assert len(result) == 1
        assert result[0] == _CellSnapshot(name="actor-0", status=NOT_APPLICABLE)

    @pytest.mark.asyncio
    async def test_http_error_raises(self) -> None:
        # [FUNCTION DOC] A mocked 500 response makes fetching cells raise the HTTP status error.
        """Verify HTTP 4xx/5xx propagated."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(status_code=500))

        with pytest.raises(httpx.HTTPStatusError):
            await runner._get_cells()


class TestRunnerPatchCell:
    # [CLASS DOC] Tests of the runner's suspend and resume patch requests (client mocked).
    @pytest.mark.asyncio
    async def test_suspend_sends_correct_patch(self) -> None:
        # [FUNCTION DOC] Suspending cell "actor-0" issues exactly one patch to the path
        # "/api/v1/cells/actor-0" whose JSON body is a spec with suspend true.
        """Verify PATCH body for suspend uses CellPatch model."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.patch = AsyncMock(return_value=_mock_response())

        await runner._suspend_cell("actor-0")

        runner._client.patch.assert_called_once()
        call_args = runner._client.patch.call_args
        assert call_args[0][0] == "/api/v1/cells/actor-0"
        body = json.loads(call_args[1]["content"])
        assert body == {"spec": {"suspend": True}}

    @pytest.mark.asyncio
    async def test_resume_sends_correct_patch(self) -> None:
        # [FUNCTION DOC] Resuming cell "actor-0" issues exactly one patch to the same path whose JSON
        # body is a spec with suspend false.
        """Verify PATCH body for resume uses CellPatch model."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.patch = AsyncMock(return_value=_mock_response())

        await runner._resume_cell("actor-0")

        runner._client.patch.assert_called_once()
        call_args = runner._client.patch.call_args
        assert call_args[0][0] == "/api/v1/cells/actor-0"
        body = json.loads(call_args[1]["content"])
        assert body == {"spec": {"suspend": False}}


class TestFtControllerDefaults:
    # [CLASS DOC] Tests of how the API server port and controller enablement are resolved from
    # command-line style arguments.
    @staticmethod
    def _resolve(**overrides) -> tuple[int, bool]:
        # [FUNCTION DOC] HELPER: Build an arguments namespace (port none, enable none, no fault
        # tolerance components) with the overrides applied, resolve and store the API server port
        # on it, then resolve controller enablement. Returns the pair of port and enable flag.
        args = argparse.Namespace(
            **{
                "api_server_port": None,
                "mini_ft_controller_enable": None,
                "ft_components": [],
                **overrides,
            }
        )
        args.api_server_port = _resolve_api_server_port(args)
        return args.api_server_port, _resolve_mini_ft_controller_enable(args)

    def test_asking_for_fault_tolerance_opens_the_port_and_starts_the_healing_loop(self) -> None:
        # [FUNCTION DOC] With the rollout fault-tolerance component and no explicit settings, the
        # result is the default port and enabled true.
        """The health checkers only publish a status; without both of these a run watches an
        engine die and leaves it routed."""
        assert self._resolve(ft_components=["rollout"]) == (_DEFAULT_FT_API_SERVER_PORT, True)

    def test_a_run_without_fault_tolerance_opens_no_port_and_starts_no_loop(self) -> None:
        # [FUNCTION DOC] With no fault-tolerance components, the result is port zero and enabled
        # false.
        """There is nothing to heal, so the port would be surface area for nobody."""
        assert self._resolve(ft_components=[]) == (0, False)

    def test_an_explicitly_disabled_port_also_disables_the_healing_loop(self) -> None:
        # [FUNCTION DOC] With the rollout component but an explicit port of zero, the result is port
        # zero and enabled false.
        """The loop drives cells over that port, so leaving it on would fail every poll."""
        assert self._resolve(ft_components=["rollout"], api_server_port=0) == (0, False)

    def test_an_explicit_port_is_kept_as_given(self) -> None:
        # [FUNCTION DOC] With the rollout component and explicit port 9999, the result is port 9999
        # and enabled true.
        """A run that pins the port has an external controller expecting to find it there."""
        assert self._resolve(ft_components=["rollout"], api_server_port=9999) == (9999, True)

    def test_an_explicit_healing_choice_wins_over_the_default(self) -> None:
        # [FUNCTION DOC] With the rollout component and the enable flag explicitly false, the enable
        # result is false (only the second element is checked).
        """The flag is how a run opts out of healing while keeping the health reporting."""
        assert self._resolve(ft_components=["rollout"], mini_ft_controller_enable=False)[1] is False


class _FakeRunner:
    # [CLASS DOC] HELPER: Fake runner recording its construction settings (API server URL, poll
    # interval, resume delay, auto-resume flag) and, when run, the properties of the thread running
    # it (daemon or not, main thread or not) plus a thread event signalling that it ran.
    def __init__(
        self, *, api_server_url: str, poll_interval: float, resume_delay: float, cells_auto_resume: bool
    ) -> None:
        # [FUNCTION DOC] HELPER: Store the settings, create an unset event, and start with unknown
        # (none) thread properties.
        self.api_server_url = api_server_url
        self.poll_interval = poll_interval
        self.resume_delay = resume_delay
        self.cells_auto_resume = cells_auto_resume
        self.ran = threading.Event()
        self.thread_was_daemon: bool | None = None
        self.thread_was_main_thread: bool | None = None

    async def run(self) -> None:
        # [FUNCTION DOC] HELPER: Record whether the current thread is a daemon and whether it is the
        # main thread, then set the ran event.
        current_thread = threading.current_thread()
        self.thread_was_daemon = current_thread.daemon
        self.thread_was_main_thread = current_thread is threading.main_thread()
        self.ran.set()


class TestMaybeStartMiniFtController:
    # [CLASS DOC] Tests of the conditional start of the controller in the background.
    def test_enabled_controller_uses_api_server_port_and_starts_a_daemon_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Replace the runner class with a spy that builds fake runners. Start with an
        # enabled arguments namespace (port 18231, poll 1.5, resume delay 2.5, kubernetes backend).
        # Assert one runner was created with URL on the loopback address and that port, the given
        # poll interval and resume delay, auto-resume true (for the kubernetes backend), and that
        # it ran (within 5 s) on a daemon thread that is not the main thread.
        """Enabling the mini FT controller aims the runner at the local API server port and runs it on a daemon thread."""
        created_runners: list[_FakeRunner] = []

        def _create_runner_spy(**kwargs: Any) -> _FakeRunner:
            # [FUNCTION DOC] HELPER: Build a fake runner from the keyword arguments, remember it,
            # and return it.
            runner = _FakeRunner(**kwargs)
            created_runners.append(runner)
            return runner

        monkeypatch.setattr(mini_ft_controller, "_MiniFTControllerRunner", _create_runner_spy)

        mini_ft_controller.maybe_start_mini_ft_controller(
            argparse.Namespace(
                mini_ft_controller_enable=True,
                api_server_port=18231,
                mini_ft_controller_poll_interval=1.5,
                mini_ft_controller_resume_delay=2.5,
                cluster_backend="kubernetes",
            )
        )

        assert len(created_runners) == 1
        runner = created_runners[0]
        assert runner.api_server_url == "http://127.0.0.1:18231"
        assert runner.poll_interval == 1.5
        assert runner.resume_delay == 2.5
        assert runner.cells_auto_resume is True
        assert runner.ran.wait(timeout=5.0)
        assert runner.thread_was_daemon is True
        assert runner.thread_was_main_thread is False
