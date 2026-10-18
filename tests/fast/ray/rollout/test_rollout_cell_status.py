# [MODULE DOC]
# Unit tests for the status reporting of rollout server cells: the phase and condition list a
# server cell derives from its state machine state and health checker verdict, the workers hash
# stamp on every status, the pending status helper used for not-yet-observed cells, and the
# controller's cell statuses mapping. Cells are real server cell objects with their private state
# and health checker overwritten, built with a stub provider that returns fixed addresses.
import time
from types import SimpleNamespace

import pytest

from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    CellState,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.server_cell import (
    INITIALIZING_TIMEOUT_SECONDS,
    ServerCell,
    ServerCellMetadata,
    compute_pending_rollout_cell_status,
)
from miles.utils.ft_utils.api_server.models import CellStatus, TriState
from miles.utils.workers.worker_spec import HostAndPort

_ADDR_INFO = CellAddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=None, gate_url="http://10.0.0.1:13000")


class _StubProvider:
    # [CLASS DOC] HELPER: Stub worker provider that gives every worker the same addresses.
    async def get_addrs(self, worker_name: str) -> dict[str, HostAndPort]:
        # [FUNCTION DOC] HELPER: Ignores the worker name and returns a mapping with a primary
        # address 10.0.0.1 port 30000 and a gate address 10.0.0.1 port 13000.
        return dict(
            primary=HostAndPort(host="10.0.0.1", port=30000),
            gate=HostAndPort(host="10.0.0.1", port=13000),
        )


def _make_cell(state: CellState, health: TriState = TriState.TRUE, workers_hash: str = "pseudo-hash-0") -> ServerCell:
    # [FUNCTION DOC] HELPER: Builds a real server cell (default test arguments, metadata for model
    # default with one GPU, no offload, weight updates on, the given workers hash, an empty router
    # client and the stub provider) and then forces its private state to the given state and its
    # private health checker to a stand-in whose status is the given tri-state verdict (default
    # true). Returns the cell; nothing is started.
    cell = ServerCell(
        args=make_args(),
        meta=ServerCellMetadata(
            model_id="default",
            worker_type="regular",
            cell_id="inference-engine-0-0-0",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            worker_name="inference-engine-0-0-0-0",
            needs_offload=False,
            update_weights=True,
            workers_hash=workers_hash,
        ),
        router_api_client=SimpleNamespace(),
        provider=_StubProvider(),
    )
    cell._state = state
    cell._health_checker = SimpleNamespace(status=health)
    return cell


def _conditions(status: CellStatus) -> list[tuple[str, TriState]]:
    # [FUNCTION DOC] HELPER: Returns the status's conditions as an ordered list of pairs of
    # condition type and tri-state value.
    return [(c.type, c.status) for c in status.conditions]


class TestServerCellStatus:
    # [CLASS DOC] Tests the phase and conditions that a server cell reports for each of its states:
    # uninitialized and initializing are Pending with only an Allocated condition (plus Healthy
    # false once initializing exceeds its deadline), pending weights is Running and healthy but
    # not serving, serving is Running with the probe verdict, and disposed is Suspended.
    def test_a_gated_cell_is_pending_without_a_health_verdict(self):
        # [FUNCTION DOC] Uninitialized cell: phase Pending, conditions only Allocated true.
        """A colocated engine waits gated for the next window; a Healthy=False here would heal-loop."""
        status = _make_cell(StateUninitialized()).cell_status()

        assert status.phase == "Pending"
        assert _conditions(status) == [("Allocated", TriState.TRUE)]

    def test_a_booting_cell_is_pending_without_a_health_verdict(self):
        # [FUNCTION DOC] Initializing cell that just started: phase Pending, conditions only
        # Allocated true.
        """Its port is not listening yet, so no probe result exists to report."""
        status = _make_cell(StateInitializing(addr_info=_ADDR_INFO, start_time=time.monotonic())).cell_status()

        assert status.phase == "Pending"
        assert _conditions(status) == [("Allocated", TriState.TRUE)]

    def test_a_cell_stuck_booting_past_its_deadline_reports_unhealthy(self):
        # [FUNCTION DOC] Initializing cell whose start time is older than the initializing timeout
        # constant: phase stays Pending, conditions are Allocated true then Healthy false.
        """Without a Healthy verdict the mini ft controller reads NotApplicable and never heals it."""
        started_long_ago = time.monotonic() - INITIALIZING_TIMEOUT_SECONDS - 1.0
        cell = _make_cell(StateInitializing(addr_info=_ADDR_INFO, start_time=started_long_ago))

        status = cell.cell_status()

        assert status.phase == "Pending"
        assert _conditions(status) == [("Allocated", TriState.TRUE), ("Healthy", TriState.FALSE)]

    def test_a_gated_cell_is_never_judged_by_the_startup_deadline(self):
        # [FUNCTION DOC] Uninitialized cell: conditions are only Allocated true.
        # KNOWN ISSUE: this repeats the first test in the class; an uninitialized cell carries no
        # start time, so nothing here exercises a deadline.
        """An uninitialized cell has not been asked to boot yet, so no startup clock is running."""
        cell = _make_cell(StateUninitialized())

        assert _conditions(cell.cell_status()) == [("Allocated", TriState.TRUE)]

    def test_a_cell_holding_stale_weights_is_already_running(self):
        # [FUNCTION DOC] Pending-weights cell: phase Running, conditions Allocated true, Healthy
        # true, Serving false (in that order).
        """It answers requests with stale weights, so a crash there is a real failure."""
        status = _make_cell(StatePendingWeights(addr_info=_ADDR_INFO)).cell_status()

        assert status.phase == "Running"
        assert _conditions(status) == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
            ("Serving", TriState.FALSE),
        ]

    def test_a_cell_holding_stale_weights_is_not_reported_as_serving(self):
        # [FUNCTION DOC] Pending-weights cell: the conditions contain Serving false.
        """Running alone cannot witness a recovery: an engine never re-registered in the router reads Running too."""
        status = _make_cell(StatePendingWeights(addr_info=_ADDR_INFO)).cell_status()

        assert ("Serving", TriState.FALSE) in _conditions(status)

    def test_a_serving_cell_reports_its_probe_verdict(self):
        # [FUNCTION DOC] Serving cell whose health checker says false: phase Running, conditions
        # Allocated true, Healthy false, Serving true.
        """This is the signal the mini ft controller heals on."""
        status = _make_cell(StateServing(addr_info=_ADDR_INFO), health=TriState.FALSE).cell_status()

        assert status.phase == "Running"
        assert _conditions(status) == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
            ("Serving", TriState.TRUE),
        ]

    def test_a_cell_registered_in_the_router_is_reported_as_serving(self):
        # [FUNCTION DOC] Serving cell with a true verdict: the conditions contain Serving true.
        """This is the only externally visible proof that a replaced engine really came back."""
        status = _make_cell(StateServing(addr_info=_ADDR_INFO)).cell_status()

        assert ("Serving", TriState.TRUE) in _conditions(status)

    def test_a_serving_cell_with_no_verdict_yet_is_unknown(self):
        # [FUNCTION DOC] Serving cell with an unknown verdict: Allocated true, Healthy unknown,
        # Serving true.
        """A checker that has not completed a probe must not be read as healthy."""
        status = _make_cell(StateServing(addr_info=_ADDR_INFO), health=TriState.UNKNOWN).cell_status()

        assert _conditions(status) == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.UNKNOWN),
            ("Serving", TriState.TRUE),
        ]

    def test_a_disposed_cell_is_suspended(self):
        # [FUNCTION DOC] Disposed cell: phase Suspended, conditions only Allocated false.
        """Nothing is left to probe once the cell has been torn down."""
        status = _make_cell(StateDisposed()).cell_status()

        assert status.phase == "Suspended"
        assert _conditions(status) == [("Allocated", TriState.FALSE)]


class TestServerCellStatusGeneration:
    # [CLASS DOC] Tests that every status is stamped with the workers hash (worker generation) of
    # the cell that computed it.
    @pytest.mark.parametrize(
        "state",
        [
            StateUninitialized(),
            StateInitializing(addr_info=_ADDR_INFO, start_time=time.monotonic()),
            StatePendingWeights(addr_info=_ADDR_INFO),
            StateServing(addr_info=_ADDR_INFO),
            StateDisposed(),
        ],
    )
    def test_a_status_is_stamped_with_the_generation_of_the_cell_that_computed_it(self, state: CellState):
        # [FUNCTION DOC] Parametrized over all five states (uninitialized, initializing, pending
        # weights, serving, disposed): a cell with workers hash hash-7 reports that hash in its
        # status.
        """No state may publish a verdict without saying which process it is about, whatever phase it reports."""
        status = _make_cell(state, workers_hash="hash-7").cell_status()

        assert status.workers_hash == "hash-7"

    def test_a_cell_stuck_booting_past_its_deadline_still_names_its_generation(self):
        # [FUNCTION DOC] Initializing cell past its startup deadline with hash hash-7: the status
        # still carries hash-7.
        """The deadline branch builds its own condition list, so it must not drop the stamp with it."""
        started_long_ago = time.monotonic() - INITIALIZING_TIMEOUT_SECONDS - 1.0
        cell = _make_cell(StateInitializing(addr_info=_ADDR_INFO, start_time=started_long_ago), workers_hash="hash-7")

        assert cell.cell_status().workers_hash == "hash-7"

    def test_two_cells_of_different_generations_report_different_stamps(self):
        # [FUNCTION DOC] Two serving cells with hashes hash-1 and hash-2 report exactly those hashes.
        """A stamp shared by every cell could not distinguish a replaced engine from its predecessor."""
        old = _make_cell(StateServing(addr_info=_ADDR_INFO), workers_hash="hash-1").cell_status()
        new = _make_cell(StateServing(addr_info=_ADDR_INFO), workers_hash="hash-2").cell_status()

        assert (old.workers_hash, new.workers_hash) == ("hash-1", "hash-2")


class TestComputePendingRolloutCellStatus:
    # [CLASS DOC] Tests the module-level helper that builds a pending status for a cell the
    # controller has not observed yet.
    @pytest.mark.parametrize("past_startup_deadline", [False, True])
    def test_a_pending_status_carries_the_generation_it_was_asked_about(self, past_startup_deadline: bool):
        # [FUNCTION DOC] Parametrized over the past-deadline flag; the returned status carries the
        # workers hash passed in (hash-7) in both cases.
        """The api server builds this itself for a cell the controller has not observed yet, from the live listing."""
        status = compute_pending_rollout_cell_status(
            workers_hash="hash-7", past_startup_deadline=past_startup_deadline
        )

        assert status.workers_hash == "hash-7"


class TestGetCellStatuses:
    # [CLASS DOC] Tests the controller's get-cell-statuses method, called unbound with a simple
    # namespace holding only a servers mapping in place of a controller.
    def _controller(self, servers: dict[str, SimpleNamespace]) -> SimpleNamespace:
        # [FUNCTION DOC] HELPER: Returns a namespace with the given servers mapping as its only
        # attribute, standing in for the controller.
        return SimpleNamespace(servers=servers)

    async def test_every_cell_of_every_server_is_reported(self):
        # [FUNCTION DOC] Two servers each with one cell (one serving, one uninitialized): the
        # result maps both cell ids to their phases, Running and Pending.
        """The api server renders one row per cell, so a missing model means missing rows."""
        cell_a = _make_cell(StateServing(addr_info=_ADDR_INFO))
        cell_b = _make_cell(StateUninitialized())
        controller = self._controller(
            {
                "actor": SimpleNamespace(server_cells={"engine-0": cell_a}),
                "critic": SimpleNamespace(server_cells={"engine-1": cell_b}),
            }
        )

        statuses = await InferenceController.get_cell_statuses(controller)

        assert {cell_id: status.phase for cell_id, status in statuses.items()} == {
            "engine-0": "Running",
            "engine-1": "Pending",
        }

    async def test_each_status_carries_the_generation_it_describes(self):
        # [FUNCTION DOC] A serving cell with hash hash-7: its status in the result carries hash-7.
        """A status without its generation is indistinguishable from one about the process it replaced."""
        controller = self._controller(
            {
                "actor": SimpleNamespace(
                    server_cells={"engine-0": _make_cell(StateServing(addr_info=_ADDR_INFO), workers_hash="hash-7")}
                )
            }
        )

        statuses = await InferenceController.get_cell_statuses(controller)

        assert statuses["engine-0"].workers_hash == "hash-7"

    async def test_a_controller_without_servers_reports_nothing(self):
        # [FUNCTION DOC] An empty servers mapping gives an empty result.
        """debug-train-only runs have no rollout cells, and must not fabricate any."""
        assert await InferenceController.get_cell_statuses(self._controller({})) == {}

    async def test_each_pending_cell_gets_its_own_status_object(self):
        # [FUNCTION DOC] Two uninitialized cells in one server: the two returned status objects are
        # distinct objects (not a shared instance).
        """A shared status instance would let one cell's mutation rewrite every other pending cell."""
        controller = self._controller(
            {
                "actor": SimpleNamespace(
                    server_cells={
                        "engine-0": _make_cell(StateUninitialized()),
                        "engine-1": _make_cell(StateUninitialized()),
                    }
                )
            }
        )

        statuses = await InferenceController.get_cell_statuses(controller)

        assert statuses["engine-0"] is not statuses["engine-1"]
