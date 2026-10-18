# [MODULE DOC] Tests for colocate pairing configuration inside generated helm values: the pairing
# config function (exactly one trainer pool required in a colocated run) and the colocate
# section of the run values built from worker specs. Verifies the section content (namespace,
# release, trainer pool id, per inference pool layout derived from GPU offsets), exclusion of
# pools placed past the trainer's GPUs, validation errors for overlapping ranges, partial-pod
# offsets, too many engine cells and sub-node cells, and the absence of the section for
# non-colocated runs or runs with no local engine.
import pydantic
import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import LAYOUT, engine, trainer

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.colocate import pairing_config
from miles.utils.workers.worker_spec import CommandWorkerSpec

COLOCATE_LAYOUT = LAYOUT.model_copy(update={"colocate": True})


def _disaggregated_engines(*, decode_offset: int) -> list[CommandWorkerSpec]:
    # [FUNCTION DOC] HELPER: Returns two 8-GPU, 2-cell engine specs named engine 0 0 (offset 0)
    # and engine 0 1 (offset equal to the given decode offset), imitating prefill and decode pools.
    return [
        engine(num_cells=2, gpus_per_engine=8, name="inference-engine-0-0", gpu_offset=0),
        engine(num_cells=2, gpus_per_engine=8, name="inference-engine-0-1", gpu_offset=decode_offset),
    ]


class TestTheValuesCarryAWholePairingConfig:
    # [CLASS DOC] Tests of the colocate section built into the run values (plus its input checks).
    @pytest.mark.parametrize("trainer_count", [0, 2])
    def test_a_colocated_run_requires_exactly_one_trainer_pool(self, trainer_count: int) -> None:
        # [FUNCTION DOC] Parametrized over zero and two trainer pools alongside one engine: the
        # pairing config call raises an assertion error stating the run has that many trainer pools.
        """A colocated run rejects both missing and ambiguous trainer pools."""
        specs = [engine(), *(trainer() for _ in range(trainer_count))]

        with pytest.raises(AssertionError, match=rf"this run has {trainer_count} trainer pools"):
            pairing_config(specs, COLOCATE_LAYOUT)

    def test_reads_the_pools_that_start_inside_the_trainer_off_their_gpu_offsets(self):
        # [FUNCTION DOC] Two engine pools (offsets 0 and 16, 8 GPUs, 2 cells) with a 4-cell, 8-GPU
        # trainer: the colocate section equals a mapping with namespace, release, trainer pool id
        # and one entry per engine pool whose layout has inference cells 2, trainer cells 4, one
        # pod per cell on each side, 8 GPUs per node, 8 GPUs per inference pod, and the pool's own
        # GPU offset.
        """The gpu offset is what already decides who shares the trainer's cards, so nothing else declares it."""
        specs = [*_disaggregated_engines(decode_offset=16), trainer(num_cells=4, gpus_per_cell=8)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert built["colocate"] == {
            "namespace": "rl",
            "release": "r",
            "trainer_pool_id": "trainer-engine-actor",
            "inference_pools": [
                {
                    "pool_id": "inference-engine-0-0",
                    "layout": {
                        "num_inference_cells": 2,
                        "num_trainer_cells": 4,
                        "num_pods_per_inference_cell": 1,
                        "num_pods_per_trainer_cell": 1,
                        "num_gpus_per_node": 8,
                        "num_gpus_per_inference_pod": 8,
                        "gpu_offset": 0,
                    },
                },
                {
                    "pool_id": "inference-engine-0-1",
                    "layout": {
                        "num_inference_cells": 2,
                        "num_trainer_cells": 4,
                        "num_pods_per_inference_cell": 1,
                        "num_pods_per_trainer_cell": 1,
                        "num_gpus_per_node": 8,
                        "num_gpus_per_inference_pod": 8,
                        "gpu_offset": 16,
                    },
                },
            ],
        }

    def test_names_the_pools_by_the_pool_id_the_pods_are_labelled_with(self):
        # [FUNCTION DOC] The colocate section's trainer pool id equals the first trainer entry's
        # pool id, and its inference pool ids equal the inference entries' pool ids in order.
        """The controller identifies a pod by its miles pool label, so the config has to speak that value."""
        specs = [*_disaggregated_engines(decode_offset=16), trainer(num_cells=4, gpus_per_cell=8)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert built["colocate"]["trainer_pool_id"] == built["trainerEngines"][0]["poolId"]
        assert [pool["pool_id"] for pool in built["colocate"]["inference_pools"]] == [
            entry["poolId"] for entry in built["inferenceEngines"]
        ]

    def test_leaves_out_a_pool_placed_past_the_trainer_gpus(self):
        # [FUNCTION DOC] The second pool at offset 32 is beyond a 2-cell, 8-GPU (16 GPU) trainer,
        # so only the first pool appears in the colocate section.
        """A prefill pool_id on its own nodes needs no pairing, and gating it would strand it forever."""
        specs = [*_disaggregated_engines(decode_offset=32), trainer(num_cells=2, gpus_per_cell=8)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert [pool["pool_id"] for pool in built["colocate"]["inference_pools"]] == ["inference-engine-0-0"]

    def test_pairs_a_pool_that_leaves_trainer_gpus_to_themselves(self):
        # [FUNCTION DOC] A 16-GPU engine pool against a single 32-GPU trainer cell is paired even
        # though it covers only part of the trainer: its pool id appears in the section.
        """Half the trainer may run no engine at all, which is still a rank-for-rank pairing where it does."""
        specs = [engine(num_cells=2, gpus_per_engine=8), trainer(num_cells=1, gpus_per_cell=32)]

        built = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

        assert [pool["pool_id"] for pool in built["colocate"]["inference_pools"]] == ["inference-engine-0-0"]

    def test_refuses_two_pools_whose_gpu_ranges_overlap(self):
        # [FUNCTION DOC] Pools at offsets 0 and 8 with 16 GPUs each overlap: building the values
        # raises a validation error saying both claim the trainer's GPU.
        """Both would be pinned onto the same node, and one of them would find no gpus left to run on."""
        specs = [*_disaggregated_engines(decode_offset=8), trainer(num_cells=4, gpus_per_cell=8)]

        with pytest.raises(pydantic.ValidationError, match="both claim the trainer's gpu"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_refuses_a_pool_starting_part_way_into_one_of_its_own_pods(self):
        # [FUNCTION DOC] An 8-GPU, one cell engine at GPU offset 4: raises a validation error
        # saying the offset is not a whole number of its own pods.
        """An 8-gpu engine starting at gpu 4 would want half of each of two nodes' worth of cards."""
        specs = [engine(num_cells=1, gpus_per_engine=8, gpu_offset=4), trainer(num_cells=4, gpus_per_cell=8)]

        with pytest.raises(pydantic.ValidationError, match="not a whole number of its own"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_refuses_more_engine_cells_than_the_trainer_can_seat(self):
        # [FUNCTION DOC] Eight 8-GPU engine cells against a 32-GPU trainer: raises a validation
        # error saying the cells do not fit.
        """An engine rank on a gpu no trainer shares would receive nothing from a weight update."""
        specs = [engine(num_cells=8, gpus_per_engine=8), trainer(num_cells=1, gpus_per_cell=32)]

        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_rejects_a_pool_whose_cell_is_smaller_than_a_node(self):
        # [FUNCTION DOC] A 4-GPU engine cell (smaller than an 8-GPU node): building raises an
        # assertion error mentioning sub-node cells.
        """The device plugin picks the cards, so a sub-node engine's base gpu id cannot be rendered."""
        specs = [engine(num_cells=1, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=4)]

        with pytest.raises(AssertionError, match="sub-node"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_places_one_engine_per_gpu_of_a_single_trainer_node(self):
        # [FUNCTION DOC] Eight 1-GPU engine cells with a single 8-GPU trainer node produce exactly
        # one inference pool entry with 8 inference cells, 1 GPU per inference pod and 8 GPUs per
        # node.
        """The shape a user reaches for first: eight engines sharing the eight cards of one trainer node."""
        specs = [engine(num_cells=8, gpus_per_engine=1), trainer(num_cells=1, gpus_per_cell=8)]

        [pool] = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]["colocate"]["inference_pools"]

        assert pool["layout"]["num_inference_cells"] == 8
        assert pool["layout"]["num_gpus_per_inference_pod"] == 1
        assert pool["layout"]["num_gpus_per_node"] == 8

    def test_refuses_a_colocated_run_whose_engines_all_sit_past_the_trainer(self):
        # [FUNCTION DOC] The only engine is at offset 32, past the trainer's GPUs: building raises
        # an assertion error saying there is nothing to pair.
        """Installing a pairing controller with nothing to pair would leave every engine gated forever."""
        specs = [engine(num_cells=1, gpus_per_engine=8, gpu_offset=32), trainer(num_cells=4, gpus_per_cell=8)]

        with pytest.raises(AssertionError, match="nothing to pair"):
            build_values(specs, COLOCATE_LAYOUT).as_values()

    def test_leaves_out_the_pairing_when_this_deployment_seats_no_engine(self):
        # [FUNCTION DOC] Only a trainer spec: the run values have no colocate key.
        """Replaying saved rollouts asks for --colocate with no local inference pool, and there is nothing to pair."""
        specs = [trainer(num_cells=4, gpus_per_cell=8)]

        assert "colocate" not in build_values(specs, COLOCATE_LAYOUT).as_values()["run"]

    def test_leaves_a_run_that_does_not_colocate_without_the_section(self):
        # [FUNCTION DOC] Using the non-colocated launch plan with engines and a trainer, the run
        # values have no colocate key.
        """A disaggregated run must not gain a pairing controller with pod write rights."""
        specs = [*_disaggregated_engines(decode_offset=16), trainer(num_cells=4, gpus_per_cell=8)]

        assert "colocate" not in build_values(specs, LAYOUT).as_values()["run"]
