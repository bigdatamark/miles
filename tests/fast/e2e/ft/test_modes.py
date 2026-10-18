# [MODULE DOC]
# Tests for the ft test mode definitions: GPU accounting of registered modes (total rollout and
# node GPUs, with colocated modes counting shared GPUs once), validation that rejects colocated
# modes whose rollout engines need more GPUs than the trainer owns, and properties of the mixed
# train-and-rollout fault-tolerance mode.
import pytest

from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode


def _mode(
    *,
    train_gpus_per_node: int,
    rollout_num_engines: int = 0,
    rollout_gpus_per_engine: int = 0,
    colocate: bool = False,
) -> FTTestMode:
    # [FUNCTION DOC] HELPER: Builds a test mode with fixed dummy model fields, two cells, empty
    # parallel args and the given trainer gpus per node, rollout engine count, gpus per engine and
    # colocate flag (defaults: no rollout, not colocated). Construction may raise an assertion
    # error for invalid combinations.
    return FTTestMode(
        model_name="model",
        model_hf_repo="org/model",
        megatron_model_type="model-type",
        num_cells=2,
        parallel_args="",
        train_gpus_per_node=train_gpus_per_node,
        rollout_num_engines=rollout_num_engines,
        rollout_gpus_per_engine=rollout_gpus_per_engine,
        colocate=colocate,
    )


class TestTotalNodeGpus:
    # [CLASS DOC] Tests for GPU totals and shape of registered modes.
    def test_rollout_only_colocated_mode_uses_plain_four_way_data_parallelism(self) -> None:
        # [FUNCTION DOC] Asserts the colocated rollout-kill mode has four cells and empty parallel args.
        """Rollout-only FT must not add context parallelism to the four-GPU trainer."""
        mode = MODES["kill_rollout__dp4__colocate"]

        assert mode.num_cells == 4
        assert mode.parallel_args == ""

    def test_colocated_mode_counts_shared_gpus_once(self) -> None:
        # [FUNCTION DOC] Asserts the colocated mode is flagged colocated, has four total rollout
        # gpus, and four total node gpus.
        """The registered colocated mode reserves only the trainer's gpus, not trainer plus rollout."""
        mode = MODES["kill_rollout__dp4__colocate"]

        assert mode.colocate
        assert mode.total_rollout_gpus == 4
        assert mode.total_node_gpus == 4

    def test_disaggregated_mode_adds_rollout_gpus_to_train_gpus(self) -> None:
        # [FUNCTION DOC] Asserts the trainer-kill mode is not colocated and uses eight node gpus in total.
        """Without colocation the rollout engines need their own gpus on top of the trainer's."""
        mode = MODES["kill_train__dp2_cp2__moe_5layer"]

        assert not mode.colocate
        assert mode.total_node_gpus == 8


class TestColocationValidation:
    # [CLASS DOC] Tests for the oversubscription check applied to colocated modes at construction.
    def test_colocated_mode_rejects_rollout_gpu_oversubscription(self) -> None:
        # [FUNCTION DOC] Builds a colocated mode with two trainer gpus and four one-gpu engines;
        # asserts an assertion error saying the mode oversubscribes its node.
        """A colocated mode whose engines need more gpus than the trainer owns is rejected at construction."""
        with pytest.raises(AssertionError, match="oversubscribes its node"):
            _mode(
                train_gpus_per_node=2,
                rollout_num_engines=4,
                rollout_gpus_per_engine=1,
                colocate=True,
            )

    def test_colocated_mode_accepts_rollout_gpus_that_exactly_fill_the_node(self) -> None:
        # [FUNCTION DOC] Four trainer gpus and four one-gpu engines, colocated; asserts construction
        # succeeds and total node gpus is four.
        """Rollout demand equal to the trainer's gpu count still fits, so construction succeeds."""
        mode = _mode(
            train_gpus_per_node=4,
            rollout_num_engines=4,
            rollout_gpus_per_engine=1,
            colocate=True,
        )

        assert mode.total_node_gpus == 4

    def test_a_disaggregated_mode_may_ask_for_more_rollout_gpus_than_train_gpus(self) -> None:
        # [FUNCTION DOC] Two trainer gpus and four one-gpu engines, not colocated; asserts
        # construction succeeds and total node gpus is six.
        """The oversubscription check applies only to colocated modes."""
        mode = _mode(
            train_gpus_per_node=2,
            rollout_num_engines=4,
            rollout_gpus_per_engine=1,
        )

        assert mode.total_node_gpus == 6


class TestMixedFtMode:
    # [CLASS DOC] Tests for the registered mixed mode that crashes both trainer and rollout cells.
    def test_the_mixed_mode_enables_ft_on_both_kinds_of_cell(self) -> None:
        # [FUNCTION DOC] Asserts the mixed mode's ft components are train and rollout and it has real rollout.
        """Trainer and rollout healing must interleave in one run, which needs ft on both."""
        mode = MODES["kill_train_rollout__dp2_cp2"]

        assert mode.ft_components == ("train", "rollout")
        assert mode.has_real_rollout

    def test_the_mixed_mode_is_disaggregated(self) -> None:
        # [FUNCTION DOC] Asserts the mixed mode is not colocated and uses eight node gpus.
        """Colocation makes a trainer crash and an engine crash contend for the same gpus, which is not the case to soak first."""
        mode = MODES["kill_train_rollout__dp2_cp2"]

        assert not mode.colocate
        assert mode.total_node_gpus == 8
