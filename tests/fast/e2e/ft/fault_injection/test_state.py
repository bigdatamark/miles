# [MODULE DOC]
# Tests for the cell-state helpers of the ft fault-injection state module: liveness of a cell
# (from its Healthy condition) and classification of a cell object into an observed cell state
# (serving versus running-but-not-serving, using the Serving condition). Inputs are fake cell
# dictionaries built by the shared test utilities.
from tests.e2e.ft.conftest_ft.fault_injection import state
from tests.fast.e2e.ft.fault_injection.utils import RUNNING_NOT_SERVING, SERVING, cell, staged


def test_cell_is_alive_true_only_when_healthy_condition_is_true() -> None:
    # [FUNCTION DOC] Asserts a cell with a true Healthy condition is alive and one with a false
    # Healthy condition is not.
    """cell_is_alive reflects the Healthy condition status."""
    assert state.cell_is_alive(cell("c", healthy=True))
    assert not state.cell_is_alive(cell("c", healthy=False))


def test_cell_is_alive_false_when_no_healthy_condition_present() -> None:
    # [FUNCTION DOC] Asserts a cell with an empty condition list is not alive.
    """A cell with no Healthy condition is not considered alive."""
    assert not state.cell_is_alive({"metadata": {"name": "c"}, "status": {"conditions": []}})


def test_a_running_cell_that_is_not_in_the_router_is_not_serving() -> None:
    # [FUNCTION DOC] Asserts a Running cell with a false Serving condition is classified as
    # running-not-serving, and one with a true Serving condition as serving.
    """The api server renders PendingWeights and Serving alike, so the Serving condition must split them."""
    assert state.compute_observed_cell_state(staged("c", RUNNING_NOT_SERVING)) is RUNNING_NOT_SERVING
    assert state.compute_observed_cell_state(staged("c", SERVING)) is SERVING
