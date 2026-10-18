# [MODULE DOC] Tests for the colocate pairing feature, which pairs each inference pod with the
# trainer pod whose node it shares (so weight updates move rank for rank) in a colocated
# Kubernetes run. Part one covers the pure layout math: the pairing layout model validation
# (positive counts, required inference pod width, no unknown fields, fit and divisibility rules),
# the equivalence of the GPU granular rules to the older pod granular ones for whole-node pools,
# the mapping from an inference pod (cell and pod index) to a target trainer pod coordinate and a
# base GPU id inside the shared node (whole node, sub node, offset and multi pod cells), and
# disjointness of the card ranges that engines on one trainer pod receive. Later parts (pod
# naming, pairing config validation, the controller and its reconcile loop against fake pod
# sources, and the helm-side support check) are documented where they appear. Module constants
# name the trainer and inference pool ids, the object name prefix and 8 GPUs per node.
from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pydantic
import pytest
from kubernetes_asyncio import client
from tests.fast.utils.workers.reconcile.utils import FakeSource, replace_of, settle

from miles.utils.external_utils.colocate_pairing import pods as pairing_pods
from miles.utils.external_utils.colocate_pairing.config import InferencePool, PairingConfig, PairingLayout
from miles.utils.external_utils.colocate_pairing.controller import (
    InferencePlacement,
    PairingController,
    _place_inference_pod,
)
from miles.utils.external_utils.colocate_pairing.pods import PodCoordinate
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.colocate import _assert_colocate_supported
from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.reconcile.source_event import DeleteEvent, UpsertEvent
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

TRAINER_POOL_ID = "trainer-engine-actor"
INFERENCE_POOL_ID = "inference-inference"
_OBJECT_NAME_PREFIX = "r-miles-run-"


GPUS_PER_NODE = 8


def _layout(
    num_inference_cells: int,
    num_trainer_cells: int,
    num_pods_per_inference_cell: int = 1,
    num_pods_per_trainer_cell: int = 1,
    gpu_offset: int = 0,
    num_gpus_per_node: int = GPUS_PER_NODE,
    num_gpus_per_inference_pod: int | None = None,
) -> PairingLayout:
    # [FUNCTION DOC] HELPER: Builds a pairing layout from cell counts, pods per cell on each side
    # (default 1), GPU offset (default 0), GPUs per node (default 8) and GPUs per inference pod
    # (default: a whole node). Validation errors of the model propagate.
    return PairingLayout(
        num_inference_cells=num_inference_cells,
        num_trainer_cells=num_trainer_cells,
        num_pods_per_inference_cell=num_pods_per_inference_cell,
        num_pods_per_trainer_cell=num_pods_per_trainer_cell,
        num_gpus_per_node=num_gpus_per_node,
        num_gpus_per_inference_pod=(
            num_gpus_per_node if num_gpus_per_inference_pod is None else num_gpus_per_inference_pod
        ),
        gpu_offset=gpu_offset,
    )


def _sub_node_layout(gpu_offset: int = 0, num_inference_cells: int = 4) -> PairingLayout:
    # [FUNCTION DOC] HELPER: Layout of half-node (4 GPU) single pod inference cells (default 4
    # cells, offset 0) against one trainer cell of 2 pods.
    return _layout(
        num_inference_cells=num_inference_cells,
        num_trainer_cells=1,
        num_pods_per_inference_cell=1,
        num_pods_per_trainer_cell=2,
        num_gpus_per_inference_pod=4,
        gpu_offset=gpu_offset,
    )


def _place(inference_cell_index: int, layout: PairingLayout, inference_pod_index: int = 0) -> InferencePlacement:
    # [FUNCTION DOC] HELPER: Calls the controller's private placement function for the inference
    # cell and pod index (default 0) with the module trainer pool id; returns the placement
    # (trainer pod coordinate and base GPU id).
    return _place_inference_pod(
        inference_cell_index=inference_cell_index,
        inference_pod_index=inference_pod_index,
        layout=layout,
        trainer_pool_id=TRAINER_POOL_ID,
    )


def _target(inference_cell_index: int, layout: PairingLayout, inference_pod_index: int = 0) -> PodCoordinate:
    # [FUNCTION DOC] HELPER: Returns the trainer pod coordinate of the placement.
    return _place(inference_cell_index, layout, inference_pod_index).trainer_coord


def _base_gpu(inference_cell_index: int, layout: PairingLayout, inference_pod_index: int = 0) -> int:
    # [FUNCTION DOC] HELPER: Returns the base GPU id (first card inside the shared node) of the placement.
    return _place(inference_cell_index, layout, inference_pod_index).base_gpu_id


def _coordinate(cell_index: int, pod_index: int = 0, pool_id: str = TRAINER_POOL_ID) -> PodCoordinate:
    # [FUNCTION DOC] HELPER: Builds a pod coordinate for the pool (default trainer), cell index and pod index in cell (default 0).
    return PodCoordinate(pool_id=pool_id, cell_index=cell_index, pod_in_cell_index=pod_index)


def _key(pool_id: str, cell_index: int, pod_index: int = 0) -> str:
    # [FUNCTION DOC] HELPER: Returns the text key of the coordinate for the pool, cell and pod.
    return _coordinate(cell_index, pod_index, pool_id).key


def _pod_name(pool_id: str, cell_index: int, pod_index: int = 0) -> str:
    # [FUNCTION DOC] HELPER: Builds the expected pod name: the object name prefix, the pool id,
    # and the cell index alone for pod 0 or the cell index, a hyphen and the pod index otherwise.
    suffix = str(cell_index) if pod_index == 0 else f"{cell_index}-{pod_index}"
    return f"{_OBJECT_NAME_PREFIX}{pool_id}-{suffix}"


class TestPairingLayout:
    # [CLASS DOC] Tests of the pairing layout model validation and derived totals.
    def test_refuses_a_pool_with_no_cells(self):
        # [FUNCTION DOC] Zero inference cells raises a validation error.
        """Zero cells is a values bug, and every later division would be by zero."""
        with pytest.raises(pydantic.ValidationError):
            _layout(num_inference_cells=0, num_trainer_cells=1)

    def test_refuses_a_cell_with_no_pods(self):
        # [FUNCTION DOC] Zero pods per inference cell raises a validation error.
        """A cell is at least one pod, and the mapping divides by the inference cell width."""
        with pytest.raises(pydantic.ValidationError):
            _layout(num_inference_cells=1, num_trainer_cells=1, num_pods_per_inference_cell=0)

    def test_refuses_a_layout_that_never_says_how_wide_an_inference_pod_is(self):
        # [FUNCTION DOC] Omitting the GPUs per inference pod field raises a validation error (it
        # is required, no default).
        """Without the pod width the pairing cannot tell a whole-node pod from four sub-node ones."""
        with pytest.raises(pydantic.ValidationError):
            PairingLayout(
                num_inference_cells=1,
                num_trainer_cells=1,
                num_pods_per_inference_cell=1,
                num_pods_per_trainer_cell=1,
                num_gpus_per_node=GPUS_PER_NODE,
                gpu_offset=0,
            )

    def test_refuses_an_inference_pod_with_no_gpus(self):
        # [FUNCTION DOC] Zero GPUs per inference pod raises a validation error.
        """A zero-gpu pod would divide by zero in the mapping and claim no gpu of the trainer's."""
        with pytest.raises(pydantic.ValidationError):
            _layout(num_inference_cells=1, num_trainer_cells=1, num_gpus_per_inference_pod=0)

    def test_counts_the_gpus_every_inference_pod_of_the_pool_holds(self):
        # [FUNCTION DOC] The sub node layout (4 cells of one 4-GPU pod) has 16 total inference GPUs.
        """The fit check is in gpus now, so the pool's width is cells times pods times the pod's gpus."""
        assert _sub_node_layout().total_inference_gpus == 16

    def test_counts_the_gpus_the_trainer_pool_holds_as_whole_nodes(self):
        # [FUNCTION DOC] The same layout (1 trainer cell of 2 pods) has 16 total trainer GPUs.
        """A trainer pod is always a whole node, which is what the inference pool is measured against."""
        assert _sub_node_layout().total_trainer_gpus == 16

    def test_refuses_an_unknown_field(self):
        # [FUNCTION DOC] An extra camel case field beyond the model's fields raises a validation error.
        """The layout comes from rendered values, so a renamed key must not be silently ignored."""
        with pytest.raises(pydantic.ValidationError):
            PairingLayout(
                num_inference_cells=1,
                num_trainer_cells=1,
                num_pods_per_inference_cell=1,
                num_pods_per_trainer_cell=1,
                num_gpus_per_node=GPUS_PER_NODE,
                num_gpus_per_inference_pod=GPUS_PER_NODE,
                gpu_offset=0,
                podsPerInferenceCell=1,
            )


_WHOLE_NODE_GRID = dict(
    num_inference_cells=(1, 2, 3),
    num_trainer_cells=(1, 2),
    num_pods_per_inference_cell=(1, 2, 3),
    num_pods_per_trainer_cell=(1, 2, 4),
)


def _whole_node_cases() -> list[dict[str, int]]:
    # [FUNCTION DOC] HELPER: Returns every combination (Cartesian product) of the grid values as
    # keyword mappings: inference cells 1-3, trainer cells 1-2, pods per inference cell 1-3 and
    # pods per trainer cell 1, 2 or 4 (54 cases).
    return [
        dict(zip(_WHOLE_NODE_GRID, values, strict=True)) for values in itertools.product(*_WHOLE_NODE_GRID.values())
    ]


def _pod_granular_layout_is_legal(*, gpu_offset: int, **case: int) -> bool:
    # [FUNCTION DOC] HELPER: Reference implementation of the old pod granular legality rules for
    # whole-node pools. Legal when: pods per inference cell does not exceed pods per trainer cell;
    # it divides it evenly; the GPU offset is a multiple of the node size; the pod offset (offset
    # divided by node size) is a multiple of pods per inference cell; and the pods from the offset
    # through all inference cells fit within all trainer pods.
    pods_per_inference_cell = case["num_pods_per_inference_cell"]
    pods_per_trainer_cell = case["num_pods_per_trainer_cell"]
    pod_offset = gpu_offset // GPUS_PER_NODE
    return (
        pods_per_inference_cell <= pods_per_trainer_cell
        and pods_per_trainer_cell % pods_per_inference_cell == 0
        and gpu_offset % GPUS_PER_NODE == 0
        and pod_offset % pods_per_inference_cell == 0
        and pod_offset + case["num_inference_cells"] * pods_per_inference_cell
        <= case["num_trainer_cells"] * pods_per_trainer_cell
    )


def _gpu_granular_layout_is_legal(*, gpu_offset: int, **case: int) -> bool:
    # [FUNCTION DOC] HELPER: Returns false if building a whole-node layout (a full node of GPUs
    # per inference pod) for the case and offset raises a validation error, otherwise true.
    try:
        _layout(gpu_offset=gpu_offset, num_gpus_per_inference_pod=GPUS_PER_NODE, **case)
    except pydantic.ValidationError:
        return False
    return True


class TestWholeNodeLayoutsAreJudgedExactlyAsTheyWereInPods:
    # [CLASS DOC] Equivalence tests between the GPU granular layout validation and the reference
    # pod granular rules.
    @pytest.mark.parametrize("gpu_offset", [0, 4, 8, 12, 16, 24, 32])
    def test_accepts_and_refuses_the_same_layouts_the_pod_granular_rules_did(self, gpu_offset: int):
        # [FUNCTION DOC] Parametrized over seven GPU offsets. For every grid case, the model's
        # verdict equals the reference verdict; the list of disagreeing cases is empty.
        """The gpu rewrite has to be a pure generalisation: a whole-node pool sees no change in verdict."""
        disagreements = [
            case
            for case in _whole_node_cases()
            if _gpu_granular_layout_is_legal(gpu_offset=gpu_offset, **case)
            != _pod_granular_layout_is_legal(gpu_offset=gpu_offset, **case)
        ]

        assert disagreements == []

    def test_the_grid_it_compares_reaches_both_verdicts(self):
        # [FUNCTION DOC] At offset 8, the grid produces both legal and illegal layouts.
        """A grid that only ever accepts, or only ever refuses, would make the comparison above vacuous."""
        verdicts = {_gpu_granular_layout_is_legal(gpu_offset=8, **case) for case in _whole_node_cases()}

        assert verdicts == {True, False}


class TestTargetTrainerPod:
    # [CLASS DOC] Tests of the mapping from an inference pod to its target trainer pod.
    def test_pairs_rank_for_rank_when_the_cells_are_the_same_width(self):
        # [FUNCTION DOC] Equal cell widths of 4 pods: inference cell 1's pod index i maps to
        # trainer cell 1, pod i.
        """Equal cell widths are the identity mapping: inference (x, y) goes to trainer (x, y)."""
        layout = _layout(
            num_inference_cells=2, num_trainer_cells=2, num_pods_per_inference_cell=4, num_pods_per_trainer_cell=4
        )

        assert [_target(1, layout, inference_pod_index=index) for index in range(4)] == [
            _coordinate(1, index) for index in range(4)
        ]

    def test_maps_a_single_pod_inference_by_dividing_and_taking_the_remainder(self):
        # [FUNCTION DOC] 8 single pod inference cells against 2 trainer cells of 4 pods: cell i maps
        # to trainer cell i divided by 4 (integer) and pod i modulo 4.
        """The one-node inference case the spec writes as (x div r, x mod r), with r inferences per trainer cell."""
        layout = _layout(
            num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
        )

        assert [_target(index, layout) for index in range(8)] == [
            _coordinate(index // 4, index % 4) for index in range(8)
        ]

    def test_tiles_several_narrow_inference_pods_across_one_trainer_cell(self):
        # [FUNCTION DOC] Same layout: the first four inference cells map to trainer cell 0, pods 0 to 3 in order.
        """A single-node inference paired with a four-node trainer cell: four inferences cover it in order."""
        layout = _layout(
            num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
        )

        assert [_target(index, layout) for index in range(4)] == [_coordinate(0, index) for index in range(4)]

    def test_moves_on_to_the_next_trainer_cell(self):
        # [FUNCTION DOC] Same layout: inference cell 4 maps to trainer cell 1, pod 0.
        """Inference five of eight belongs to the second trainer cell, not the first."""
        layout = _layout(
            num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
        )

        assert _target(4, layout) == _coordinate(1)

    def test_pairs_a_two_node_inference_with_half_a_four_node_trainer_cell(self):
        # [FUNCTION DOC] 4 inference cells of 2 pods against 2 trainer cells of 4 pods: inference
        # cell 1's two pods map to trainer cell 0, pods 2 and 3.
        """The general case 1 < K_e < K_t: two inference pods land on two of the trainer cell's four."""
        layout = _layout(
            num_inference_cells=4, num_trainer_cells=2, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=4
        )

        assert [_target(1, layout, inference_pod_index=index) for index in range(2)] == [
            _coordinate(0, 2),
            _coordinate(0, 3),
        ]

    def test_seats_two_sub_node_inference_pods_on_one_trainer_pod(self):
        # [FUNCTION DOC] With the half-node layout, cells 0 to 3 map to trainer pods (0,0), (0,0),
        # (0,1), (0,1): two inference pods per trainer pod.
        """Half-node inference pods share the node their trainer pod holds, two of them per trainer pod."""
        layout = _sub_node_layout()

        assert [_target(index, layout) for index in range(4)] == [
            _coordinate(0, 0),
            _coordinate(0, 0),
            _coordinate(0, 1),
            _coordinate(0, 1),
        ]

    def test_seats_four_quarter_node_inference_pods_on_one_trainer_pod(self):
        # [FUNCTION DOC] 8 inference cells of 2-GPU pods against one trainer cell of 2 pods: cell i
        # maps to trainer pod i divided by 4 (four per trainer pod).
        """The pod width alone decides how many inference pods a trainer's node seats, down to a quarter."""
        layout = _layout(
            num_inference_cells=8,
            num_trainer_cells=1,
            num_pods_per_inference_cell=1,
            num_pods_per_trainer_cell=2,
            num_gpus_per_inference_pod=2,
        )

        assert [_target(index, layout) for index in range(8)] == [_coordinate(0, index // 4) for index in range(8)]

    def test_refuses_an_inference_wider_than_a_trainer_cell(self):
        # [FUNCTION DOC] Inference cell of 4 pods against a trainer cell of 2 raises a validation
        # error containing the phrase cannot fit.
        """K_e > K_t: its extra ranks would have no trainer node to sit on, so colocate cannot hold."""
        with pytest.raises(pydantic.ValidationError, match="cannot fit"):
            _layout(
                num_inference_cells=1, num_trainer_cells=1, num_pods_per_inference_cell=4, num_pods_per_trainer_cell=2
            )

    def test_refuses_inference_cells_that_do_not_divide_a_trainer_cell(self):
        # [FUNCTION DOC] Inference cells of 3 pods against a trainer cell of 4 raises a validation
        # error mentioning a whole number.
        """One inference would straddle two trainer cells, and its ranks would disagree about their peer."""
        with pytest.raises(pydantic.ValidationError, match="whole number"):
            _layout(
                num_inference_cells=2, num_trainer_cells=1, num_pods_per_inference_cell=3, num_pods_per_trainer_cell=4
            )

    def test_refuses_a_pool_larger_than_the_trainer_can_seat(self):
        # [FUNCTION DOC] 3 single pod inference cells against 2 trainer cells of 1 pod raises a
        # validation error saying they do not fit.
        """The third inference has no trainer cell left, so its weight update would transfer nothing."""
        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            _layout(
                num_inference_cells=3, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=1
            )

    def test_refuses_an_inference_index_outside_the_pool(self):
        # [FUNCTION DOC] Placing inference cell 9 in a 2-cell layout raises an assertion error
        # saying it is outside.
        """A stale pod from a shrunk release must not be paired against arithmetic that no longer holds."""
        with pytest.raises(AssertionError, match="outside"):
            _target(9, _layout(num_inference_cells=2, num_trainer_cells=2))

    def test_refuses_a_pod_index_outside_its_cell(self):
        # [FUNCTION DOC] Pod index 5 in a cell of 2 pods raises an assertion error saying outside.
        """A worker index beyond the cell width means the name was parsed against the wrong pool_id."""
        layout = _layout(
            num_inference_cells=2, num_trainer_cells=2, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=2
        )

        with pytest.raises(AssertionError, match="outside"):
            _target(0, layout, inference_pod_index=5)


class TestBaseGpuIdOfAnInferencePod:
    # [CLASS DOC] Tests of the base GPU id (first card inside the shared node) of placements.
    def test_a_whole_node_pod_starts_at_the_first_card(self):
        # [FUNCTION DOC] Whole node pods of a 2-cell layout both have base GPU id 0.
        """Its pod holds every card of the node, so the node-local numbering starts where the node does."""
        layout = _layout(num_inference_cells=2, num_trainer_cells=2)

        assert [_base_gpu(index, layout) for index in range(2)] == [0, 0]

    def test_sub_node_pods_tile_the_cards_of_the_node_they_share(self):
        # [FUNCTION DOC] Half-node layout base GPU ids across the four cells are 0, 4, 0, 4.
        """Two half-node pods on one trainer node must not both claim the cards the node numbers from zero."""
        layout = _sub_node_layout()

        assert [_base_gpu(index, layout) for index in range(4)] == [0, 4, 0, 4]

    def test_one_gpu_pods_walk_the_whole_node(self):
        # [FUNCTION DOC] 8 one-GPU inference cells against one single pod trainer cell have base
        # GPU ids 0 to 7 in order.
        """The recipe default is one gpu per engine, and eight of them cover a node card by card."""
        layout = _layout(
            num_inference_cells=8, num_trainer_cells=1, num_pods_per_trainer_cell=1, num_gpus_per_inference_pod=1
        )

        assert [_base_gpu(index, layout) for index in range(8)] == list(range(8))

    def test_the_pods_of_a_wide_engine_each_start_at_their_own_node(self):
        # [FUNCTION DOC] One inference cell of 2 whole-node pods: both pods have base GPU id 0.
        """An engine wider than a node is one whole-node pod per node, and each is handed its cards from zero."""
        layout = _layout(
            num_inference_cells=1,
            num_trainer_cells=1,
            num_pods_per_inference_cell=2,
            num_pods_per_trainer_cell=2,
        )

        assert [_base_gpu(0, layout, inference_pod_index=index) for index in range(2)] == [0, 0]

    def test_the_pods_of_a_wide_engine_pair_with_adjacent_trainer_pods(self):
        # [FUNCTION DOC] The same layout: the two pods map to trainer cell 0, pods 0 and 1.
        """Its two pods sit on two nodes, so they must wait on the two trainer pods holding those nodes."""
        layout = _layout(
            num_inference_cells=1,
            num_trainer_cells=1,
            num_pods_per_inference_cell=2,
            num_pods_per_trainer_cell=2,
        )

        assert [_target(0, layout, inference_pod_index=index) for index in range(2)] == [
            _coordinate(0, 0),
            _coordinate(0, 1),
        ]

    def test_the_pool_offset_moves_the_first_card_along(self):
        # [FUNCTION DOC] Half-node layout with 3 cells and GPU offset 4 has base GPU ids 4, 0, 4.
        """A second pool starts part way into the node its neighbour shares, and its offset says how far."""
        layout = _sub_node_layout(gpu_offset=4, num_inference_cells=3)

        assert [_base_gpu(index, layout) for index in range(3)] == [4, 0, 4]

    def test_a_multi_pod_cell_of_sub_node_pods_walks_across_the_trainer_pods(self):
        # [FUNCTION DOC] 2 cells of 2 half-node pods each against one trainer cell of 2 pods: the
        # four placements in cell-major order are (trainer pod 0, GPU 0), (0, 4), (1, 0), (1, 4).
        """Cells and pods per cell were only ever varied one at a time; together they index one flat gpu line."""
        layout = _layout(
            num_inference_cells=2,
            num_trainer_cells=1,
            num_pods_per_inference_cell=2,
            num_pods_per_trainer_cell=2,
            num_gpus_per_inference_pod=4,
        )
        placements = [_place(cell, layout, pod) for cell in range(2) for pod in range(2)]

        assert [(placement.trainer_coord, placement.base_gpu_id) for placement in placements] == [
            (_coordinate(0, 0), 0),
            (_coordinate(0, 0), 4),
            (_coordinate(0, 1), 0),
            (_coordinate(0, 1), 4),
        ]

    def test_a_multi_pod_cell_of_quarter_node_pods_stays_on_one_trainer_pod(self):
        # [FUNCTION DOC] 2 cells of 2 two-GPU pods: all four placements share trainer pod (0,0)
        # and have base GPU ids 0, 2, 4, 6.
        """Four quarter-node pods spread over two cells still share one node, a pair of its cards each."""
        layout = _layout(
            num_inference_cells=2,
            num_trainer_cells=1,
            num_pods_per_inference_cell=2,
            num_pods_per_trainer_cell=2,
            num_gpus_per_inference_pod=2,
        )
        placements = [_place(cell, layout, pod) for cell in range(2) for pod in range(2)]

        assert {placement.trainer_coord for placement in placements} == {_coordinate(0, 0)}
        assert [placement.base_gpu_id for placement in placements] == [0, 2, 4, 6]


_DISJOINT_CARD_LAYOUTS = {
    "half node pods, two cells of two": _layout(
        num_inference_cells=2,
        num_trainer_cells=1,
        num_pods_per_inference_cell=2,
        num_pods_per_trainer_cell=2,
        num_gpus_per_inference_pod=4,
    ),
    "quarter node pods on one trainer pod": _layout(
        num_inference_cells=2,
        num_trainer_cells=1,
        num_pods_per_inference_cell=2,
        num_pods_per_trainer_cell=2,
        num_gpus_per_inference_pod=2,
    ),
    "one gpu pods across two nodes": _layout(
        num_inference_cells=16,
        num_trainer_cells=2,
        num_pods_per_trainer_cell=1,
        num_gpus_per_inference_pod=1,
    ),
    "half node pods offset by one pod": _sub_node_layout(gpu_offset=4, num_inference_cells=3),
}


def _cards_by_trainer_pod(layouts: dict[str, PairingLayout]) -> dict[PodCoordinate, list[tuple[str, range]]]:
    # [FUNCTION DOC] HELPER: For every pool layout, places every pod of every cell and groups the
    # results by trainer pod coordinate. Each entry is the pool id and the range of card numbers
    # from the base GPU id for the pod's GPU count. Returns the mapping.
    cards: dict[PodCoordinate, list[tuple[str, range]]] = {}
    for pool_id, layout in layouts.items():
        for cell_index in range(layout.num_inference_cells):
            for pod_index in range(layout.num_pods_per_inference_cell):
                placement = _place(cell_index, layout, pod_index)
                span = range(placement.base_gpu_id, placement.base_gpu_id + layout.num_gpus_per_inference_pod)
                cards.setdefault(placement.trainer_coord, []).append((pool_id, span))
    return cards


def _cards_claimed_by_two_trainer_pods(cards: dict[PodCoordinate, list[tuple[str, range]]]) -> list[Any]:
    # [FUNCTION DOC] HELPER: Returns tuples of (trainer pod, first entry, second entry) for every
    # pair of entries on the same trainer pod whose card ranges overlap; empty when all disjoint.
    # (The name says two trainer pods but the overlap is between two inference pods.)
    return [
        (trainer, first, second)
        for trainer, entries in cards.items()
        for first, second in itertools.combinations(entries, 2)
        if set(first[1]) & set(second[1])
    ]


class TestTheEnginesOfOneTrainerPodHoldDifferentCards:
    # [CLASS DOC] Tests that engines sharing a trainer pod get disjoint card ranges.
    @pytest.mark.parametrize("name", sorted(_DISJOINT_CARD_LAYOUTS))
    def test_no_two_pods_of_a_pool_are_given_an_overlapping_stretch(self, name: str):
        # [FUNCTION DOC] Parametrized over four named layouts. The overlap list for the layout is empty.
        """Two engines on one node sharing a card is the failure this whole mechanism exists to avoid."""
        cards = _cards_by_trainer_pod({name: _DISJOINT_CARD_LAYOUTS[name]})

        assert _cards_claimed_by_two_trainer_pods(cards) == []

    @pytest.mark.parametrize("name", sorted(_DISJOINT_CARD_LAYOUTS))
    def test_at_least_one_trainer_pod_really_seats_several_of_them(self, name: str):
        # [FUNCTION DOC] Same layouts: at least one trainer pod holds two or more inference pods,
        # so the disjointness test is not vacuous.
        """A layout whose trainer pods seat one engine each would satisfy the test above without testing it."""
        cards = _cards_by_trainer_pod({name: _DISJOINT_CARD_LAYOUTS[name]})

        assert max(len(entries) for entries in cards.values()) >= 2

    def test_two_pools_splitting_a_node_are_given_different_halves_of_it(self):
        # [FUNCTION DOC] Two pools (the inference pool at offset 0 and a decode pool at offset 4),
        # one half-node cell each: no overlapping cards, and trainer pod (0,0) holds exactly two entries.
        """Prefill and decode share a trainer node on purpose, and only their offsets keep them apart."""
        cards = _cards_by_trainer_pod(
            {
                INFERENCE_POOL_ID: _sub_node_layout(num_inference_cells=1),
                DECODE_POOL_ID: _sub_node_layout(gpu_offset=4, num_inference_cells=1),
            }
        )

        assert _cards_claimed_by_two_trainer_pods(cards) == []
        assert len(cards[_coordinate(0, 0)]) == 2


def _target_by_cell_division(
    inference_cell_index: int, inference_pod_index: int, layout: PairingLayout
) -> PodCoordinate:
    # [FUNCTION DOC] HELPER: Reference pod granular mapping for whole node pools: inferences per
    # trainer cell is trainer pods divided by inference pods per cell; the trainer cell is the
    # inference cell divided by that count; the pod is the remainder times pods per inference cell
    # plus the inference pod index. Returns the trainer coordinate.
    inferences_per_trainer_cell = layout.num_pods_per_trainer_cell // layout.num_pods_per_inference_cell
    trainer_cell_index = inference_cell_index // inferences_per_trainer_cell
    offset_within_cell = inference_cell_index % inferences_per_trainer_cell
    return _coordinate(
        trainer_cell_index, offset_within_cell * layout.num_pods_per_inference_cell + inference_pod_index
    )


def _all_targets(layout: PairingLayout) -> list[PodCoordinate]:
    # [FUNCTION DOC] HELPER: Target trainer coordinates of every pod of every inference cell, in cell-major order.
    return [
        _target(cell_index, layout, inference_pod_index=pod_index)
        for cell_index in range(layout.num_inference_cells)
        for pod_index in range(layout.num_pods_per_inference_cell)
    ]


class TestGpuOffsetPairing:
    # [CLASS DOC] Tests of how a pool's GPU offset shifts its target trainer pods.
    @pytest.mark.parametrize(
        "layout",
        [
            _layout(
                num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
            ),
            _layout(
                num_inference_cells=4, num_trainer_cells=2, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=4
            ),
            _layout(
                num_inference_cells=2, num_trainer_cells=2, num_pods_per_inference_cell=4, num_pods_per_trainer_cell=4
            ),
        ],
        ids=["narrow", "half-cell", "whole-cell"],
    )
    def test_a_pool_at_the_first_gpu_maps_exactly_as_dividing_by_cells_did(self, layout):
        # [FUNCTION DOC] Parametrized over three zero-offset layouts (narrow, half-cell,
        # whole-cell). All targets equal those from the reference cell-division mapping.
        """The offset form has to be a generalisation, so at offset zero it must reproduce the old mapping."""
        assert _all_targets(layout) == [
            _target_by_cell_division(cell_index, pod_index, layout)
            for cell_index in range(layout.num_inference_cells)
            for pod_index in range(layout.num_pods_per_inference_cell)
        ]

    def test_an_offset_pool_starts_at_the_trainer_pod_holding_its_first_gpu(self):
        # [FUNCTION DOC] 2 single pod cells at GPU offset 16 against 4 trainer pods in one cell
        # map to trainer pods 2 and 3.
        """The offset is what makes a second pool land beside the trainers it actually reads weights from."""
        layout = _layout(
            num_inference_cells=2,
            num_trainer_cells=1,
            num_pods_per_inference_cell=1,
            num_pods_per_trainer_cell=4,
            gpu_offset=16,
        )

        assert _all_targets(layout) == [_coordinate(0, 2), _coordinate(0, 3)]

    def test_an_offset_crosses_into_the_next_trainer_cell(self):
        # [FUNCTION DOC] One cell at offset 16 against 2 trainer cells of 2 pods maps to trainer cell 1, pod 0.
        """Trainer cells are numbered off the same flat gpu range, so an offset can carry a pool into the next."""
        layout = _layout(
            num_inference_cells=1,
            num_trainer_cells=2,
            num_pods_per_inference_cell=1,
            num_pods_per_trainer_cell=2,
            gpu_offset=16,
        )

        assert _all_targets(layout) == [_coordinate(1)]

    def test_refuses_an_offset_that_starts_inside_a_node(self):
        # [FUNCTION DOC] A whole-node pool at offset 4 raises a validation error saying the offset
        # is not a whole number of its own pods.
        """Half a node is not a whole-node pod, and the inference would want gpus of two trainer pods at once."""
        with pytest.raises(pydantic.ValidationError, match="is not a whole number of its own"):
            _layout(num_inference_cells=1, num_trainer_cells=2, num_pods_per_trainer_cell=2, gpu_offset=4)

    def test_refuses_an_offset_that_starts_inside_a_sub_node_pod(self):
        # [FUNCTION DOC] A half-node pool at offset 2 raises the same validation error.
        """The offset is counted in the pool's own pods, so two gpus in is half a four-gpu pod, not a start."""
        with pytest.raises(pydantic.ValidationError, match="is not a whole number of its own"):
            _sub_node_layout(gpu_offset=2, num_inference_cells=1)

    def test_starts_a_sub_node_pool_on_the_trainer_pod_holding_the_offset_gpu(self):
        # [FUNCTION DOC] Half-node pool of 3 cells at offset 4 maps to trainer pods (0,0), (0,1), (0,1).
        """A pool offset by one sub-node pod still lands on the trainer pod whose node holds that gpu."""
        layout = _sub_node_layout(gpu_offset=4, num_inference_cells=3)

        assert _all_targets(layout) == [_coordinate(0, 0), _coordinate(0, 1), _coordinate(0, 1)]

    def test_refuses_an_offset_that_splits_an_inference_cell_across_trainer_cells(self):
        # [FUNCTION DOC] A 2-pod cell at offset 8 (one pod) against trainer cells of 4 pods raises a
        # validation error saying its cells would straddle trainer cells.
        """An offset of half a cell leaves the pool unaligned, so a later cell of it would span two trainer cells."""
        with pytest.raises(pydantic.ValidationError, match="so its cells would straddle trainer cells"):
            _layout(
                num_inference_cells=1,
                num_trainer_cells=2,
                num_pods_per_inference_cell=2,
                num_pods_per_trainer_cell=4,
                gpu_offset=8,
            )

    def test_allows_an_offset_of_whole_inference_cells(self):
        # [FUNCTION DOC] The same layout at offset 16 (two pods) is accepted and maps to trainer pods 2 and 3 of cell 0.
        """Offsetting by a whole two-pod cell keeps every later cell inside one trainer cell."""
        layout = _layout(
            num_inference_cells=1,
            num_trainer_cells=2,
            num_pods_per_inference_cell=2,
            num_pods_per_trainer_cell=4,
            gpu_offset=16,
        )

        assert _all_targets(layout) == [_coordinate(0, 2), _coordinate(0, 3)]

    def test_refuses_a_pool_that_the_offset_pushes_past_the_last_trainer_pod(self):
        # [FUNCTION DOC] 2 cells at offset 8 against 2 trainer pods raises a validation error
        # saying they do not fit.
        """Its last inference would sit on a gpu no trainer holds, and a weight update would transfer nothing."""
        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            _layout(
                num_inference_cells=2,
                num_trainer_cells=1,
                num_pods_per_inference_cell=1,
                num_pods_per_trainer_cell=2,
                gpu_offset=8,
            )


class TestLayoutPairs:
    # [CLASS DOC] Tests of cell level legality of layouts.
    def test_allows_trainer_cells_that_seat_no_inference(self):
        # [FUNCTION DOC] 2 inference cells with 4 trainer cells builds without error.
        """A prefill pool_id on its own nodes leaves trainer gpus with no inference, which is a legal run."""
        _layout(num_inference_cells=2, num_trainer_cells=4)

    def test_allows_a_trainer_cell_seating_fewer_inference_pods_than_it_could(self):
        # [FUNCTION DOC] 2 single pod inference cells with one 4-pod trainer cell builds without error.
        """Half a trainer cell may run inferences and the other half none, which is still rank-for-rank."""
        _layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4)

    def test_refuses_more_inference_cells_than_the_trainer_cells_seat(self):
        # [FUNCTION DOC] 8 single pod inference cells against one 4-pod trainer cell raises a
        # validation error saying they do not fit.
        """The surplus inferences would pair with a trainer cell the run never created."""
        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            _layout(
                num_inference_cells=8, num_trainer_cells=1, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
            )

    def test_refuses_an_inference_cell_that_straddles_two_trainer_cells(self):
        # [FUNCTION DOC] A 2-pod inference cell against a 3-pod trainer cell raises a validation
        # error mentioning a whole number.
        """No single trainer cell then owns the inference, so healing one would leave the other half live."""
        with pytest.raises(pydantic.ValidationError, match="whole number"):
            _layout(
                num_inference_cells=1, num_trainer_cells=1, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=3
            )


class TestGpuWidthPairs:
    # [CLASS DOC] Tests of GPU width legality of layouts.
    def test_refuses_an_inference_pod_that_does_not_divide_a_node(self):
        # [FUNCTION DOC] 3 GPUs per inference pod on 8-GPU nodes raises a validation error saying
        # the width does not divide a node.
        """Three gpus of an eight-gpu node means a later pod of the pool would straddle two trainer pods."""
        with pytest.raises(pydantic.ValidationError, match="does not divide a"):
            _layout(num_inference_cells=1, num_trainer_cells=1, num_gpus_per_inference_pod=3)

    def test_refuses_an_inference_pod_wider_than_a_node(self):
        # [FUNCTION DOC] 16 GPUs per inference pod raises the same validation error.
        """A pod wider than the node it is pinned to would need gpus the trainer pod beside it holds."""
        with pytest.raises(pydantic.ValidationError, match="does not divide a"):
            _layout(num_inference_cells=1, num_trainer_cells=2, num_gpus_per_inference_pod=16)

    def test_refuses_sub_node_pods_that_overrun_the_trainer_gpus_by_less_than_a_node(self):
        # [FUNCTION DOC] 4 half-node cells at offset 4 overrun the 16 trainer GPUs by 4: raises a
        # validation error saying they do not fit in the trainer's GPUs.
        """Counting in pods would round this down and miss it: the last half-node pod has no trainer gpu."""
        with pytest.raises(pydantic.ValidationError, match="do not fit in the trainer's"):
            _sub_node_layout(gpu_offset=4, num_inference_cells=4)

    def test_allows_sub_node_pods_that_exactly_fill_the_trainer_gpus(self):
        # [FUNCTION DOC] 4 half-node cells at offset 0 fit exactly: 16 total inference GPUs.
        """The fit check is inclusive, so a pool covering every trainer gpu is the legal maximum."""
        assert _sub_node_layout(num_inference_cells=4).total_inference_gpus == 16


class TestAssertColocateSupported:
    # [CLASS DOC] Tests of the helm side support check (imported private function) taking GPUs per
    # node, GPUs per inference pod and GPUs per trainer pod.
    def test_accepts_whole_node_cells_that_tile(self):
        # [FUNCTION DOC] 8, 8 and 8 are accepted without error.
        """What the launcher checks before rendering: whole-node pods and an inference pool_id that tiles."""
        _assert_colocate_supported(
            num_gpus_per_node=GPUS_PER_NODE,
            gpus_per_inference_pod=8,
            gpus_per_trainer_pod=8,
        )

    def test_accepts_a_sub_node_inference_cell_and_pairs_it_with_the_trainer_pod_holding_its_gpus(self):
        # [FUNCTION DOC] 4-GPU inference pods on 8-GPU trainer pods are accepted, and the half-node
        # layout's targets are trainer pods (0,0), (0,0), (0,1), (0,1).
        """Several small inference pods may share one trainer node, each paired with the pod running there."""
        _assert_colocate_supported(
            num_gpus_per_node=GPUS_PER_NODE,
            gpus_per_inference_pod=4,
            gpus_per_trainer_pod=8,
        )

        assert _all_targets(_sub_node_layout()) == [
            _coordinate(0, 0),
            _coordinate(0, 0),
            _coordinate(0, 1),
            _coordinate(0, 1),
        ]

    def test_refuses_an_inference_pod_larger_than_a_node(self):
        # [FUNCTION DOC] 16 GPUs per inference pod raises an assertion error saying it reaches past
        # the node it is pinned to.
        """It would be pinned to one node while holding gpus of the next, which colocate cannot honour."""
        with pytest.raises(AssertionError, match="reaches past the node it is pinned to"):
            _assert_colocate_supported(
                num_gpus_per_node=GPUS_PER_NODE,
                gpus_per_inference_pod=16,
                gpus_per_trainer_pod=8,
            )

    def test_refuses_a_sub_node_trainer_pod(self):
        # [FUNCTION DOC] 4 GPUs per trainer pod raises an assertion error mentioning a sub-node pod.
        """The device plugin hands the trainer arbitrary cards, so only a whole-node one owns every index."""
        with pytest.raises(AssertionError, match="sub-node pod"):
            _assert_colocate_supported(
                num_gpus_per_node=GPUS_PER_NODE,
                gpus_per_inference_pod=8,
                gpus_per_trainer_pod=4,
            )


class TestPoolsClaimDistinctGpus:
    # [CLASS DOC] Tests that a pairing config rejects pools claiming the same trainer GPU.
    def test_refuses_two_pools_that_want_the_same_trainer_gpus(self):
        # [FUNCTION DOC] Two whole-node pools of 2 cells at offsets 0 and 8 overlap: config
        # construction raises a validation error saying both claim the trainer's gpu 8.
        """Only one inference can hold a node's gpus, and the second would land on gpus already taken."""
        with pytest.raises(pydantic.ValidationError, match="both claim the trainer's gpu 8"):
            _config(
                [
                    _inference_pool(_layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4)),
                    _inference_pool(
                        _layout(
                            num_inference_cells=2,
                            num_trainer_cells=1,
                            num_pods_per_trainer_cell=4,
                            gpu_offset=8,
                        ),
                        pool_id=DECODE_POOL_ID,
                    ),
                ]
            )

    def test_refuses_two_sub_node_pools_that_want_the_same_gpu_of_one_node(self):
        # [FUNCTION DOC] A 2-cell half-node pool and a 1-cell one at offset 4 overlap on GPU 4:
        # config construction raises a validation error naming gpu 4.
        """Sub-node pools share a node on purpose, so the overlap check has to be per gpu, not per node."""
        with pytest.raises(pydantic.ValidationError, match="both claim the trainer's gpu 4"):
            _config(
                [
                    _inference_pool(_sub_node_layout(num_inference_cells=2)),
                    _inference_pool(_sub_node_layout(gpu_offset=4, num_inference_cells=1), pool_id=DECODE_POOL_ID),
                ]
            )

    def test_allows_two_sub_node_pools_that_split_one_node_between_them(self):
        # [FUNCTION DOC] One-cell half-node pools at offsets 0 and 4 build a config without error.
        """Two half-node pools on one trainer node is the point of sub-node cells, not a collision."""
        _config(
            [
                _inference_pool(_sub_node_layout(num_inference_cells=1)),
                _inference_pool(_sub_node_layout(gpu_offset=4, num_inference_cells=1), pool_id=DECODE_POOL_ID),
            ]
        )


BASE_GPU_ID_ANNOTATION = DEFAULT_LABEL_KEYS.base_gpu_id_annotation
_ESCAPED_BASE_GPU_ID_ANNOTATION = "miles.radixark.io~1base-gpu-id"
_CHART_META_ANNOTATION = f"{DEFAULT_LABEL_KEYS.meta_annotation_prefix}{DEFAULT_LABEL_KEYS.gpu_ids_meta}"
_POD_ANNOTATIONS = {_CHART_META_ANNOTATION: "0", "leaderworkerset.sigs.k8s.io/size": "2"}


def _release_patch(
    *,
    node_name: str = "gpu-7",
    base_gpu_id: int = 0,
    gates: list[str] | None = None,
    annotations: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Builds the release patch (list of json patch operations) via the pods
    # module. Defaults: node name gpu-7, base GPU id 0, the pairing gate as the only scheduling
    # gate, and the module's pod annotations (chart gpu meta plus a size annotation).
    return pairing_pods.release_patch(
        node_name=node_name,
        base_gpu_id=base_gpu_id,
        gates=[pairing_pods._GATE_NAME] if gates is None else gates,
        annotations=_POD_ANNOTATIONS if annotations is None else annotations,
    )


class TestReleasePatch:
    # [CLASS DOC] Tests of the json patch that pins a gated pod to a node, hands it its first card
    # as an annotation and removes the pairing scheduling gate.
    def test_pins_the_pod_to_one_node_and_removes_the_gate(self):
        # [FUNCTION DOC] The first operation adds a required node affinity matching the metadata
        # name field In the single node name; the operations from index 2 are a test of the gate
        # name at gate index 0 then a removal of gate 0.
        """Both in one patch, so a controller restart cannot leave a pinned pod still gated."""
        patch = _release_patch()

        assert patch[0] == {
            "op": "add",
            "path": "/spec/affinity",
            "value": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {"matchFields": [{"key": "metadata.name", "operator": "In", "values": ["gpu-7"]}]}
                        ]
                    }
                }
            },
        }
        assert patch[2:] == [
            {"op": "test", "path": "/spec/schedulingGates/0/name", "value": pairing_pods._GATE_NAME},
            {"op": "remove", "path": "/spec/schedulingGates/0"},
        ]

    def test_tells_the_pod_which_card_of_the_node_is_its_own(self):
        # [FUNCTION DOC] For base GPU id 5 the second operation adds the escaped base GPU id
        # annotation path (slash escaped as tilde one) with the text value 5.
        """The pod cannot work this out itself, and the same patch that seats it is what hands it over."""
        patch = _release_patch(base_gpu_id=5)

        assert patch[1] == {
            "op": "add",
            "path": f"/metadata/annotations/{_ESCAPED_BASE_GPU_ID_ANNOTATION}",
            "value": "5",
        }

    def test_writes_the_card_as_a_string_the_downward_api_can_serve(self):
        # [FUNCTION DOC] The single operation touching metadata for base GPU id 7 has the text value 7.
        """A fieldRef reads an annotation, and an annotation value that is not a string is not valid."""
        [operation] = [op for op in _release_patch(base_gpu_id=7) if str(op["path"]).startswith("/metadata")]

        assert operation["value"] == "7"

    def test_never_replaces_the_map_of_a_pod_that_carries_annotations(self):
        # [FUNCTION DOC] The annotation operation targets the single key path and no operation
        # targets the whole annotations map.
        """A whole-map add replaces it, dropping the chart's gpu meta and the platform's own bookkeeping."""
        patch = _release_patch(base_gpu_id=3, annotations=_POD_ANNOTATIONS)

        assert patch[1]["path"] == f"/metadata/annotations/{_ESCAPED_BASE_GPU_ID_ANNOTATION}"
        assert not any(operation["path"] == "/metadata/annotations" for operation in patch)

    def test_refuses_to_build_a_patch_for_a_pod_that_carries_no_annotations(self):
        # [FUNCTION DOC] Empty annotations raise an assertion error saying the pod carries no annotations.
        """The apiserver would reject the add anyway; failing here names the pod state that caused it."""
        with pytest.raises(AssertionError, match="carries no annotations"):
            _release_patch(base_gpu_id=3, annotations={})

    def test_pins_by_node_name_rather_than_by_hostname_label(self):
        # [FUNCTION DOC] The affinity has exactly one term, a match on fields (the metadata name),
        # not on labels.
        """A node whose hostname label differs from its name would match no node at all."""
        [term] = _release_patch()[0]["value"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
            "nodeSelectorTerms"
        ]

        assert term == {"matchFields": [{"key": "metadata.name", "operator": "In", "values": ["gpu-7"]}]}

    def test_removes_only_its_own_gate(self):
        # [FUNCTION DOC] With gates other, pairing, other, the test and remove operations address gate index 1.
        """Dropping the whole list would release a pod another controller is still deliberately holding back."""
        gates = ["other.io/first", pairing_pods._GATE_NAME, "other.io/last"]

        patch = _release_patch(gates=gates)

        assert patch[2:] == [
            {"op": "test", "path": "/spec/schedulingGates/1/name", "value": pairing_pods._GATE_NAME},
            {"op": "remove", "path": "/spec/schedulingGates/1"},
        ]

    def test_is_a_json_patch_rather_than_a_merge(self):
        # [FUNCTION DOC] Every element of the patch has an operation key.
        """A merge patch setting the gates to an empty list is silently ignored: the list merges by name."""
        patch = _release_patch()

        assert all("op" in operation for operation in patch)

    def test_hands_over_the_card_only_if_the_gate_is_still_there(self):
        # [FUNCTION DOC] The operation kinds are, in order, add, add, test, remove (the test guards
        # the removal; the annotation add precedes the guard in the list).
        # KNOWN ISSUE: the name and docstring say the test guards the whole patch, but the two
        # adds come before the test operation; json patch is atomic, so a failed test still aborts
        # all of them, which is what actually provides the guarantee.
        """A test op guards the whole patch, so a pod another pass released is never re-annotated."""
        patch = _release_patch(base_gpu_id=5)

        assert [operation["op"] for operation in patch] == ["add", "add", "test", "remove"]


class TestCoordinateOf:
    # [CLASS DOC] Tests of reading a pod's coordinate from its labels.
    def test_reads_the_pool_and_the_indices_the_chart_labelled(self):
        # [FUNCTION DOC] A pod labelled with the inference pool, cell 2 and pod 3 yields that coordinate.
        """The same three labels the worker provider reads, so both sides identify a pod the same way."""
        assert pairing_pods.coordinate_of(_pod(INFERENCE_POOL_ID, 2, 3)) == _coordinate(2, 3, INFERENCE_POOL_ID)

    def test_does_not_read_the_name(self):
        # [FUNCTION DOC] Renaming the pod does not change the coordinate read from its labels.
        """A pod name is the release's object name plus indices; only the labels are the contract."""
        renamed = _pod(INFERENCE_POOL_ID, 2, 3)
        renamed.metadata.name = "something-else-entirely"

        assert pairing_pods.coordinate_of(renamed) == _coordinate(2, 3, INFERENCE_POOL_ID)

    def test_returns_none_for_a_pod_the_run_did_not_label(self):
        # [FUNCTION DOC] A pod with empty labels yields none.
        """Every pod of the release comes down one stream, and most of them belong to no pool."""
        assert pairing_pods.coordinate_of(_unlabelled_pod("r-miles-run-orchestrator-0")) is None


def _pod(
    pool_id: str,
    cell_index: int = 0,
    pod_index: int = 0,
    *,
    node_name: str | None = None,
    gated: bool = True,
    node_selector: Any = None,
) -> Any:
    # [FUNCTION DOC] HELPER: Builds a fake pod object (simple namespaces) named by the standard
    # naming for the pool, cell and pod index, then sets labels for the pool id, cell index and pod
    # in cell index, and an annotation for the chart's gpu meta with value 0. Keyword options: node
    # name (default none), gated (default true: carries the pairing scheduling gate) and node
    # selector (default none).
    pod = _unlabelled_pod(
        _pod_name(pool_id, cell_index, pod_index),
        node_name=node_name,
        gated=gated,
        node_selector=node_selector,
    )
    pod.metadata.labels = {
        DEFAULT_LABEL_KEYS.pool_id: pool_id,
        DEFAULT_LABEL_KEYS.cell_index: str(cell_index),
        DEFAULT_LABEL_KEYS.pod_in_cell_index: str(pod_index),
    }
    pod.metadata.annotations = {f"{DEFAULT_LABEL_KEYS.meta_annotation_prefix}{DEFAULT_LABEL_KEYS.gpu_ids_meta}": "0"}
    return pod


def _unlabelled_pod(name: str, *, node_name: str | None = None, gated: bool = True, node_selector: Any = None) -> Any:
    # [FUNCTION DOC] HELPER: Builds a fake pod (nested simple namespaces) with the given name, a
    # uid derived from the name, empty labels and annotations, no deletion timestamp, the given
    # node name, a scheduling gates list holding the pairing gate when gated (else empty), the
    # node selector, no subdomain, and an empty status (no ip, conditions or container statuses).
    gates = [SimpleNamespace(name=pairing_pods._GATE_NAME)] if gated else []
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, uid=f"uid-{name}", labels={}, annotations={}, deletion_timestamp=None),
        spec=SimpleNamespace(node_name=node_name, scheduling_gates=gates, node_selector=node_selector, subdomain=None),
        status=SimpleNamespace(pod_ip=None, conditions=[], container_statuses=[]),
    )


class FakeLoop:
    # [CLASS DOC] HELPER: Fake reconcile loop that serves a fixed list of pods for any parent key.
    def __init__(self, pods: list[Any]) -> None:
        # [FUNCTION DOC] HELPER: Keeps a copy of the pod list.
        self._pods = list(pods)

    def get_by_parent(self, parent_key: str) -> list[Any]:
        # [FUNCTION DOC] HELPER: Returns a copy of all pods regardless of the parent key.
        return list(self._pods)


def _attached(controller: PairingController, pods: list[Any]) -> PairingController:
    # [FUNCTION DOC] HELPER: Gives the controller a fake loop serving the pods, and returns the controller.
    controller.set_loop(FakeLoop(pods))
    return controller


def _inference_pool(layout: PairingLayout, pool_id: str = INFERENCE_POOL_ID) -> InferencePool:
    # [FUNCTION DOC] HELPER: Wraps the layout into an inference pool with the pool id (default the module's).
    return InferencePool(pool_id=pool_id, layout=layout)


def _config(pools: list[InferencePool]) -> PairingConfig:
    # [FUNCTION DOC] HELPER: Builds a pairing config for namespace rl, release r, the module trainer pool id and the pools.
    return PairingConfig(namespace="rl", release="r", trainer_pool_id=TRAINER_POOL_ID, inference_pools=pools)


def _controller(core_v1: Any, layout: PairingLayout | None = None) -> PairingController:
    # [FUNCTION DOC] HELPER: Builds a pairing controller over the given fake core API with one
    # inference pool using the layout (default 2 inference cells against 2 trainer cells).
    pools = [_inference_pool(layout or _layout(num_inference_cells=2, num_trainer_cells=2))]
    return PairingController(config=_config(pools), core_v1=core_v1)


def _base_gpu_id_written(body: list[dict[str, Any]]) -> str:
    # [FUNCTION DOC] HELPER: Finds the single patch operation touching the annotations path and
    # returns its value, taking the base GPU id key out of it when the value is a mapping (a whole
    # map write) rather than text.
    [operation] = [op for op in body if str(op["path"]).startswith("/metadata/annotations")]
    value = operation["value"]
    return value if isinstance(value, str) else value[BASE_GPU_ID_ANNOTATION]


class TestReconcile:
    # [CLASS DOC] Tests of the controller's reconcile for one trainer pod key, using a fake core
    # API that records patched pods (name and patch) and can reject chosen pods with a status
    # code. The loop supplies the pods; each test runs the asynchronous reconcile to completion.
    def test_releases_a_gated_inference_onto_its_trainer_node(self):
        # [FUNCTION DOC] A gated inference pod 0 and a running trainer pod 0 on node gpu-3:
        # exactly one patch, for inference pod 0, equal to the release patch for node gpu-3.
        """This is the whole point: the inference ends up where the trainer that feeds it already runs."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 0)))

        assert core_v1.patched == [
            (
                _pod_name(INFERENCE_POOL_ID, 0),
                _release_patch(node_name="gpu-3"),
            )
        ]

    def test_releases_every_sub_node_inference_the_trainer_seats_in_one_pass(self):
        # [FUNCTION DOC] Half-node layout: reconciling trainer pod (0,0) patches inference pods 0
        # and 1 with node gpu-3 and base GPU ids 0 and 4.
        """One trainer node wakes both half-node inference pods, and one reconcile has to place them both."""
        core_v1 = FakeCoreV1()
        pods = [
            _pod(INFERENCE_POOL_ID, 0),
            _pod(INFERENCE_POOL_ID, 1),
            _pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False),
        ]

        asyncio.run(_attached(_controller(core_v1, _sub_node_layout()), pods).reconcile(_key(TRAINER_POOL_ID, 0, 0)))

        assert core_v1.patched == [
            (
                _pod_name(INFERENCE_POOL_ID, index),
                _release_patch(node_name="gpu-3", base_gpu_id=index * 4),
            )
            for index in (0, 1)
        ]

    def test_gives_every_engine_of_a_shared_node_a_card_of_its_own(self):
        # [FUNCTION DOC] Eight one-GPU inference pods and one trainer pod: the base GPU ids written
        # across the patches are 0 to 7 as text, in order.
        """Eight one-gpu engines on one trainer node is the shape this exists for, and 0..7 is the answer."""
        core_v1 = FakeCoreV1()
        layout = _layout(
            num_inference_cells=8,
            num_trainer_cells=1,
            num_pods_per_trainer_cell=1,
            num_gpus_per_inference_pod=1,
        )
        pods = [_pod(INFERENCE_POOL_ID, index) for index in range(8)]
        pods.append(_pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False))

        asyncio.run(_attached(_controller(core_v1, layout), pods).reconcile(_key(TRAINER_POOL_ID, 0)))

        assert [_base_gpu_id_written(body) for _, body in core_v1.patched] == [str(index) for index in range(8)]

    def test_gives_a_whole_node_engine_the_first_card(self):
        # [FUNCTION DOC] Inference pod 1 with trainer pod 1 on a node: the written base GPU id is 0.
        """The engine holds every card of the node, so it is handed them from zero and is told exactly that."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, 1), _pod(TRAINER_POOL_ID, 1, node_name="gpu-3", gated=False)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 1)))

        assert [_base_gpu_id_written(body) for _, body in core_v1.patched] == ["0"]

    def test_starts_a_second_sub_node_pool_where_its_own_offset_says(self):
        # [FUNCTION DOC] Two half-node pools (offsets 0 and 4) both seat on trainer pod (0,0):
        # the written base GPU ids are 0 for the first pool and 4 for the second.
        """Two half-node pools split one trainer node, and only the offset tells the second one apart."""
        core_v1 = FakeCoreV1()
        controller = PairingController(
            config=_config(
                [
                    _inference_pool(_sub_node_layout(num_inference_cells=1)),
                    _inference_pool(_sub_node_layout(gpu_offset=4, num_inference_cells=1), pool_id=DECODE_POOL_ID),
                ]
            ),
            core_v1=core_v1,
        )
        pods = [
            _pod(INFERENCE_POOL_ID, 0),
            _pod(DECODE_POOL_ID, 0),
            _pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False),
        ]

        asyncio.run(_attached(controller, pods).reconcile(_key(TRAINER_POOL_ID, 0, 0)))

        assert [_base_gpu_id_written(body) for _, body in core_v1.patched] == ["0", "4"]

    def test_releases_only_the_inference_pods_of_the_trainer_being_reconciled(self):
        # [FUNCTION DOC] Four inference pods with the half-node layout, reconciling trainer pod
        # (0,1) on node gpu-4: only inference pods 2 and 3 are patched.
        """The other trainer pod's node is a different machine, so its inference pods must stay gated."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, index) for index in range(4)]
        pods.append(_pod(TRAINER_POOL_ID, 0, 1, node_name="gpu-4", gated=False))

        asyncio.run(_attached(_controller(core_v1, _sub_node_layout()), pods).reconcile(_key(TRAINER_POOL_ID, 0, 1)))

        assert [name for name, _ in core_v1.patched] == [_pod_name(INFERENCE_POOL_ID, index) for index in (2, 3)]

    def test_releases_the_still_gated_pod_when_its_neighbour_is_already_placed(self):
        # [FUNCTION DOC] Inference pod 0 is already placed (ungated); only still-gated pod 1 is patched.
        """A partly released trainer node is the normal retry state, and the gate left is the only work."""
        core_v1 = FakeCoreV1()
        pods = [
            _pod(INFERENCE_POOL_ID, 0, node_name="gpu-3", gated=False),
            _pod(INFERENCE_POOL_ID, 1),
            _pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False),
        ]

        asyncio.run(_attached(_controller(core_v1, _sub_node_layout()), pods).reconcile(_key(TRAINER_POOL_ID, 0, 0)))

        assert [name for name, _ in core_v1.patched] == [_pod_name(INFERENCE_POOL_ID, 1)]

    def test_logs_a_rejected_patch_instead_of_aborting_the_pass(self, caplog):
        # [FUNCTION DOC] The fake API rejects the patch for inference pod 0 with status 500. The
        # reconcile does not raise, nothing is recorded as patched, and an error level log contains
        # the fake apiserver's rejection text.
        """Raising leaves the gates behind it untouched, and the loop is at-least-once: log and go on."""
        core_v1 = FakeCoreV1(rejects={_pod_name(INFERENCE_POOL_ID, 0): 500})
        pods = [
            _pod(INFERENCE_POOL_ID, 0),
            _pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False),
        ]

        with caplog.at_level(logging.ERROR):
            asyncio.run(
                _attached(_controller(core_v1, _sub_node_layout()), pods).reconcile(_key(TRAINER_POOL_ID, 0, 0))
            )

        assert core_v1.patched == []
        assert "rejected by the fake apiserver" in caplog.text

    def test_releases_the_neighbour_of_a_pod_whose_patch_was_refused(self):
        # [FUNCTION DOC] Pod 0's patch is rejected with status 422; pod 1 is still patched.
        """A pod released by an earlier pass is refused, and the gate behind it is the work of this pass."""
        core_v1 = FakeCoreV1(rejects={_pod_name(INFERENCE_POOL_ID, 0): 422})
        pods = [
            _pod(INFERENCE_POOL_ID, 0),
            _pod(INFERENCE_POOL_ID, 1),
            _pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False),
        ]

        asyncio.run(_attached(_controller(core_v1, _sub_node_layout()), pods).reconcile(_key(TRAINER_POOL_ID, 0, 0)))

        assert [name for name, _ in core_v1.patched] == [_pod_name(INFERENCE_POOL_ID, 1)]

    def test_keeps_a_selector_the_pod_already_carries(self):
        # [FUNCTION DOC] An inference pod with a node selector is patched, and the first operation
        # adds the spec affinity (the selector is not removed or replaced).
        """The run's global nodeSelector is on the pod, and removing it makes the apiserver refuse."""
        inference = _pod(INFERENCE_POOL_ID, 0, node_selector={"pool": "gpu"})
        core_v1 = FakeCoreV1()
        pods = [inference, _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 0)))

        assert core_v1.patched[0][1][0]["path"] == "/spec/affinity"

    def test_waits_while_the_trainer_has_no_node(self):
        # [FUNCTION DOC] The trainer pod has no node name yet: no pod is patched.
        """Releasing now would let the scheduler put the inference anywhere, which is the bug gates prevent."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_waits_while_the_trainer_does_not_exist_yet(self):
        # [FUNCTION DOC] Only a gated inference pod exists (no trainer pod): no pod is patched.
        """helm creates both pool_ids at once, so an inference routinely reconciles before its trainer appears."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, 0)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_does_nothing_for_an_inference_already_released(self):
        # [FUNCTION DOC] Both pods already ungated and on gpu-3: no pod is patched.
        """Removing a gate cannot be undone, so a released pod is terminal and patching again is noise."""
        core_v1 = FakeCoreV1()
        pods = [
            _pod(INFERENCE_POOL_ID, 0, node_name="gpu-3", gated=False),
            _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False),
        ]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_does_nothing_for_an_inference_that_disappeared(self):
        # [FUNCTION DOC] With an empty pod list, reconciling a trainer key patches nothing.
        """A scaled-down release deletes pods, and their queued reconciles must not resurrect anything."""
        core_v1 = FakeCoreV1()
        pods = []

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_is_idempotent_when_run_twice(self):
        # [FUNCTION DOC] Reconciles twice; between runs the inference pod's gates are emptied (as
        # the cluster would after the patch). Only one patch was recorded in total.
        """Delivery is at-least-once, and the second run sees a pod that is no longer gated."""
        inference = _pod(INFERENCE_POOL_ID, 0)
        core_v1 = FakeCoreV1()
        pods = [inference, _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)]
        controller = _attached(_controller(core_v1), pods)

        asyncio.run(controller.reconcile(_key(TRAINER_POOL_ID, 0)))
        inference.spec.scheduling_gates = []
        asyncio.run(controller.reconcile(_key(TRAINER_POOL_ID, 0)))

        assert len(core_v1.patched) == 1


class TestKeyOf:
    # [CLASS DOC] Tests of the controller's key function, which maps a pod to the reconcile key
    # (a trainer pod key, or a key with the unrelated prefix for pods that pair with nothing).
    def test_keys_an_inference_pod_by_the_trainer_it_waits_on(self):
        # [FUNCTION DOC] Inference pod 1 gets the key of trainer pod (1, 0).
        """The trainer's node is the only thing it waits for, so both pods have to reach one key."""
        controller = _controller(FakeCoreV1())

        assert controller.key_of(_pod(INFERENCE_POOL_ID, 1)) == _key(TRAINER_POOL_ID, 1)

    def test_keys_a_trainer_pod_by_itself(self):
        # [FUNCTION DOC] Trainer pod 1 gets its own key.
        """A trainer getting a node is the event its inference pods wait on, and the pair is keyed by it."""
        controller = _controller(FakeCoreV1())

        assert controller.key_of(_pod(TRAINER_POOL_ID, 1)) == _key(TRAINER_POOL_ID, 1)

    def test_keys_every_sub_node_inference_pod_of_one_node_by_the_same_trainer(self):
        # [FUNCTION DOC] With the half-node layout, inference pods 0 to 3 get the keys of trainer
        # pods (0,0), (0,0), (0,1), (0,1).
        """The loop hands a pod exactly one key, so a trainer seating several inference pods needs one key."""
        controller = _controller(FakeCoreV1(), _sub_node_layout())

        assert [controller.key_of(_pod(INFERENCE_POOL_ID, index)) for index in range(4)] == [
            _key(TRAINER_POOL_ID, 0, 0),
            _key(TRAINER_POOL_ID, 0, 0),
            _key(TRAINER_POOL_ID, 0, 1),
            _key(TRAINER_POOL_ID, 0, 1),
        ]

    def test_keys_a_trainer_that_seats_no_inference_apart(self):
        # [FUNCTION DOC] With 1 inference cell and 4 trainer cells, trainer pod 3 seats nothing and
        # its key starts with the unrelated prefix.
        """A trainer cell the run left empty must not be routed to an inference key nothing waits on."""
        controller = _controller(FakeCoreV1(), _layout(num_inference_cells=1, num_trainer_cells=4))

        assert controller.key_of(_pod(TRAINER_POOL_ID, 3)).startswith("__unrelated__/")

    def test_keys_an_unrelated_pod_apart(self):
        # [FUNCTION DOC] A pod without pool labels (the orchestrator) gets a key with the unrelated prefix.
        """The orchestrator shares the release label and must not be mistaken for anything pairable."""
        controller = _controller(FakeCoreV1())

        assert controller.key_of(_unlabelled_pod("r-miles-run-orchestrator-0")).startswith("__unrelated__/")


DECODE_POOL_ID = "inference-inference-decode"


def _two_pool_controller(core_v1: Any) -> PairingController:
    # [FUNCTION DOC] HELPER: Builds a controller with two whole-node pools of 2 cells each against
    # one trainer cell of 4 pods: the main inference pool at offset 0 and the decode pool at
    # offset 16 (pods 2 and 3).
    pools = [
        _inference_pool(_layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4)),
        _inference_pool(
            _layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4, gpu_offset=16),
            pool_id=DECODE_POOL_ID,
        ),
    ]
    return PairingController(config=_config(pools), core_v1=core_v1)


class TestSeveralInferencePools:
    # [CLASS DOC] Tests of a controller driving several colocated inference pools.
    def test_each_pool_lands_on_the_trainer_pods_its_own_offset_names(self):
        # [FUNCTION DOC] Main pool pod 1 keys to trainer pod (0,1); decode pool pod 0 keys to (0,2).
        """One controller drives every colocated pool_id, and the offset is all that tells them apart."""
        controller = _two_pool_controller(FakeCoreV1())

        assert [controller.key_of(_pod(INFERENCE_POOL_ID, 1)), controller.key_of(_pod(DECODE_POOL_ID, 0))] == [
            _key(TRAINER_POOL_ID, 0, 1),
            _key(TRAINER_POOL_ID, 0, 2),
        ]

    def test_releases_a_pod_of_the_second_pool_onto_its_own_trainer(self):
        # [FUNCTION DOC] Decode pod 1 with trainer pod (0,3) on node gpu-4: reconciling that key
        # patches the decode pod with the release patch for gpu-4.
        """A prefill/decode run has two gated pool_ids, and reconcile has to know which layout a pod follows."""
        core_v1 = FakeCoreV1()
        pods = [_pod(DECODE_POOL_ID, 1), _pod(TRAINER_POOL_ID, 0, 3, node_name="gpu-4", gated=False)]

        asyncio.run(_attached(_two_pool_controller(core_v1), pods).reconcile(_key(TRAINER_POOL_ID, 0, 3)))

        assert core_v1.patched == [
            (
                _pod_name(DECODE_POOL_ID, 1),
                _release_patch(node_name="gpu-4"),
            )
        ]

    def test_keys_every_pool_by_the_trainer_pod_its_layout_names(self):
        # [FUNCTION DOC] Decode pod 1 keys to trainer pod (0,3).
        """key_of runs over one stream of pods, so a pod of either pool_id has to reach its own trainer."""
        controller = _two_pool_controller(FakeCoreV1())

        assert controller.key_of(_pod(DECODE_POOL_ID, 1)) == _key(TRAINER_POOL_ID, 0, 3)

    def test_seats_two_sub_node_pools_side_by_side_on_one_trainer_pod(self):
        # [FUNCTION DOC] Two half-node pools at offsets 0 and 4: pod 0 of each keys to trainer pod (0,0).
        """Prefill and decode may split a trainer node, and both then wait on that one trainer pod."""
        controller = PairingController(
            config=_config(
                [
                    _inference_pool(_sub_node_layout(num_inference_cells=1)),
                    _inference_pool(_sub_node_layout(gpu_offset=4, num_inference_cells=1), pool_id=DECODE_POOL_ID),
                ]
            ),
            core_v1=FakeCoreV1(),
        )

        assert [controller.key_of(_pod(INFERENCE_POOL_ID, 0)), controller.key_of(_pod(DECODE_POOL_ID, 0))] == [
            _key(TRAINER_POOL_ID, 0, 0),
            _key(TRAINER_POOL_ID, 0, 0),
        ]

    def test_releases_both_sub_node_pools_of_one_trainer_pod_together(self):
        # [FUNCTION DOC] Same two pools with gated pods of each and a placed trainer pod: one
        # reconcile patches the main pool's pod then the decode pool's pod, in that order.
        """The two pools share a node, so the trainer landing there has to place a pod of each."""
        core_v1 = FakeCoreV1()
        controller = PairingController(
            config=_config(
                [
                    _inference_pool(_sub_node_layout(num_inference_cells=1)),
                    _inference_pool(_sub_node_layout(gpu_offset=4, num_inference_cells=1), pool_id=DECODE_POOL_ID),
                ]
            ),
            core_v1=core_v1,
        )
        pods = [
            _pod(INFERENCE_POOL_ID, 0),
            _pod(DECODE_POOL_ID, 0),
            _pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False),
        ]

        asyncio.run(_attached(controller, pods).reconcile(_key(TRAINER_POOL_ID, 0, 0)))

        assert [name for name, _ in core_v1.patched] == [
            _pod_name(INFERENCE_POOL_ID, 0),
            _pod_name(DECODE_POOL_ID, 0),
        ]


class TestPairingConfig:
    # [CLASS DOC] Tests of the pairing config model's json handling and validation.
    def test_reads_back_what_the_chart_serialised(self):
        # [FUNCTION DOC] A config with one decode pool dumped to json and validated back equals the original.
        """The chart passes one json blob, and a field lost in transit would pair pods with the wrong trainer."""
        config = PairingConfig(
            namespace="rl",
            release="run",
            trainer_pool_id=TRAINER_POOL_ID,
            inference_pools=[
                InferencePool(
                    pool_id=DECODE_POOL_ID,
                    layout=_layout(
                        num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4, gpu_offset=16
                    ),
                )
            ],
        )

        restored = PairingConfig.model_validate_json(config.model_dump_json())

        assert restored == config

    def test_refuses_a_config_that_names_no_pool(self):
        # [FUNCTION DOC] An empty list of inference pools raises a validation error.
        """A pairing controller with nothing to pair would sit there while every inference pod stays gated."""
        with pytest.raises(pydantic.ValidationError):
            PairingConfig(namespace="rl", release="run", trainer_pool_id=TRAINER_POOL_ID, inference_pools=[])

    def test_refuses_a_layout_the_chart_should_never_render(self):
        # [FUNCTION DOC] Json whose layout has an inference cell of 4 pods against a trainer cell
        # of 2 raises a validation error containing cannot fit on parsing.
        """The json crosses a process boundary, so the invariants have to be rechecked on the reading side."""
        payload = (
            '{"namespace": "rl", "release": "run", "trainer_pool_id": "t", "inference_pools": '
            '[{"pool_id": "d", "layout": {"num_inference_cells": 1, "num_trainer_cells": 1, '
            '"num_pods_per_inference_cell": 4, "num_pods_per_trainer_cell": 2, "num_gpus_per_node": 8, '
            '"num_gpus_per_inference_pod": 8, "gpu_offset": 0}}]}'
        )

        with pytest.raises(pydantic.ValidationError, match="cannot fit"):
            PairingConfig.model_validate_json(payload)

    def test_refuses_a_layout_whose_sub_node_pods_overrun_the_trainer(self):
        # [FUNCTION DOC] Json with 5 half-node cells against one trainer cell of 2 pods raises a
        # validation error saying they do not fit in the trainer's GPUs.
        """The gpu-level fit check has to survive the json boundary too, or the surplus pod waits forever."""
        payload = (
            '{"namespace": "rl", "release": "run", "trainer_pool_id": "t", "inference_pools": '
            '[{"pool_id": "d", "layout": {"num_inference_cells": 5, "num_trainer_cells": 1, '
            '"num_pods_per_inference_cell": 1, "num_pods_per_trainer_cell": 2, "num_gpus_per_node": 8, '
            '"num_gpus_per_inference_pod": 4, "gpu_offset": 0}}]}'
        )

        with pytest.raises(pydantic.ValidationError, match="do not fit in the trainer's"):
            PairingConfig.model_validate_json(payload)


class FakeCoreV1:
    # [CLASS DOC] HELPER: Fake Kubernetes core API recording pod patches, with optional per-pod
    # failures (a mapping from pod name to the status code to reject with).
    def __init__(self, *, rejects: dict[str, int] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Starts with no recorded patches; stores the reject mapping (empty by default).
        self.patched: list[tuple[str, list[dict[str, Any]]]] = []
        self._rejects = rejects or {}

    async def patch_namespaced_pod(self, *, name: str, namespace: str, body: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC] HELPER: If the pod name is in the reject mapping, raises the Kubernetes
        # client's API exception with that status and a rejected-by-the-fake-apiserver reason;
        # otherwise records the (name, patch body) pair. The namespace is ignored.
        if (status := self._rejects.get(name)) is not None:
            raise client.ApiException(status=status, reason="rejected by the fake apiserver")
        self.patched.append((name, body))


class PairingHarness:
    # [CLASS DOC] HELPER: Test harness wiring a pairing controller into a real reconcile loop over
    # a fake pod source, a fake clock (resync every 60 seconds) and the fake core API. Tests drive
    # it through events and read the recorded patches.
    def __init__(self, *, layout: PairingLayout | None = None, rejects: dict[str, int] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Builds the fake core API (with optional rejections), source and
        # clock, a controller with one pool (default layout 2 cells against 2 trainer cells), and
        # the loop using the controller's reconcile and key functions; attaches the loop to the
        # controller.
        self.core_v1 = FakeCoreV1(rejects=rejects)
        self.source = FakeSource()
        self.clock = FakeClock()
        pools = [_inference_pool(layout or _layout(num_inference_cells=2, num_trainer_cells=2))]
        self.controller = PairingController(config=_config(pools), core_v1=self.core_v1)
        self.loop = ReconcileLoop(
            source=self.source,
            reconcile=self.controller.reconcile,
            key_map=self.controller.key_of,
            clock=self.clock,
            resync_period=60.0,
        )
        self.controller.set_loop(self.loop)

    @asynccontextmanager
    async def running(self, *pods: Any) -> AsyncIterator[PairingHarness]:
        # [FUNCTION DOC] HELPER: Asynchronous context manager. Starts the loop as a task, lets it
        # settle, emits the pods as the initial full listing, settles, waits for start to finish,
        # and yields the harness. On exit (even on error) it stops the loop.
        start_task = asyncio.create_task(self.loop.start())
        await settle()
        self.source.emit(replace_of(*pods))
        await settle()
        await start_task
        try:
            yield self
        finally:
            await self.loop.stop()

    async def upsert(self, pod: Any) -> None:
        # [FUNCTION DOC] HELPER: Emits an upsert event keyed by the pod name and lets the loop settle.
        self.source.emit(UpsertEvent(key=pod.metadata.name, obj=pod))
        await settle()

    async def delete(self, pod: Any) -> None:
        # [FUNCTION DOC] HELPER: Emits a delete event keyed by the pod name (carrying the last
        # object) and lets the loop settle.
        self.source.emit(DeleteEvent(key=pod.metadata.name, last_obj=pod))
        await settle()

    def patched_names(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the names of patched pods in patch order.
        return [name for name, _ in self.core_v1.patched]


class TestEventSequences:
    # [CLASS DOC] Asynchronous end to end tests through the harness: event sequences drive the
    # reconcile loop and the recorded patches are checked.
    async def test_releases_the_inference_when_its_trainer_lands_afterwards(self):
        # [FUNCTION DOC] Initially inference pod 0 and an unplaced trainer pod 0: no patch. After
        # an upsert placing the trainer on gpu-3, exactly one patch for inference pod 0 with the
        # gpu-3 release patch.
        """Inference first: nothing happens until the trainer's own event brings the reconcile back."""
        harness = PairingHarness()

        async with harness.running(_pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0)):
            assert harness.core_v1.patched == []

            await harness.upsert(_pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False))

            assert harness.core_v1.patched == [
                (
                    _pod_name(INFERENCE_POOL_ID, 0),
                    _release_patch(node_name="gpu-3"),
                ),
            ]

    async def test_releases_the_inference_when_the_trainer_was_there_first(self):
        # [FUNCTION DOC] Initial placed trainer pod 1 on gpu-9; upserting gated inference pod 1
        # produces exactly one patch with the gpu-9 release patch.
        """Trainer first: the inference's own arrival finds a placed trainer and releases immediately."""
        harness = PairingHarness()

        async with harness.running(_pod(TRAINER_POOL_ID, 1, node_name="gpu-9", gated=False)):
            await harness.upsert(_pod(INFERENCE_POOL_ID, 1))

            assert harness.core_v1.patched == [
                (
                    _pod_name(INFERENCE_POOL_ID, 1),
                    _release_patch(node_name="gpu-9"),
                ),
            ]

    async def test_releases_every_sub_node_inference_when_their_shared_trainer_lands(self):
        # [FUNCTION DOC] Half-node layout with two inference pods and an unplaced trainer pod (0,0):
        # no patch; after the trainer is placed on gpu-3, both inference pods are patched in order.
        """One trainer event has to carry both half-node inference pods onto that node, not just one."""
        harness = PairingHarness(layout=_sub_node_layout())

        async with harness.running(
            _pod(INFERENCE_POOL_ID, 0), _pod(INFERENCE_POOL_ID, 1), _pod(TRAINER_POOL_ID, 0, 0)
        ):
            assert harness.core_v1.patched == []

            await harness.upsert(_pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False))

            assert harness.patched_names() == [
                _pod_name(INFERENCE_POOL_ID, 0),
                _pod_name(INFERENCE_POOL_ID, 1),
            ]

    async def test_leaves_the_inference_gated_while_the_trainer_has_no_node(self):
        # [FUNCTION DOC] An upsert of a trainer pod still without a node leaves no patches.
        """A trainer pod exists long before it is scheduled, and its node is the only thing worth waiting on."""
        harness = PairingHarness()

        async with harness.running(_pod(INFERENCE_POOL_ID, 0)):
            await harness.upsert(_pod(TRAINER_POOL_ID, 0))

            assert harness.core_v1.patched == []

    async def test_does_not_patch_an_inference_that_is_already_released(self):
        # [FUNCTION DOC] Both pods ungated and on gpu-3 in the initial listing: no patch.
        """A controller restart relists a world where the gate is gone, which is a terminal state."""
        harness = PairingHarness()

        async with harness.running(
            _pod(INFERENCE_POOL_ID, 0, node_name="gpu-3", gated=False),
            _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False),
        ):
            assert harness.core_v1.patched == []

    async def test_does_not_patch_twice_when_the_release_is_observed_back(self):
        # [FUNCTION DOC] After the initial listing causes one patch, an upsert showing the
        # inference pod released does not add another: still only that one pod name patched.
        """The patch produces its own watch event, and re-entering must not repeat the write."""
        harness = PairingHarness()

        async with harness.running(
            _pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)
        ):
            await harness.upsert(_pod(INFERENCE_POOL_ID, 0, node_name="gpu-3", gated=False))

            assert harness.patched_names() == [_pod_name(INFERENCE_POOL_ID, 0)]

    async def test_re_enters_safely_after_a_crash_before_the_patch_landed(self):
        # [FUNCTION DOC] Two separate harnesses are started with the same initial pods (simulating
        # a restarted controller); the second one also patches the inference pod once.
        """A controller that died before patching sees the same gated pod again and finishes the job."""
        crashed = PairingHarness()
        async with crashed.running(
            _pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)
        ):
            pass

        restarted = PairingHarness()
        async with restarted.running(
            _pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)
        ):
            assert restarted.patched_names() == [_pod_name(INFERENCE_POOL_ID, 0)]

    async def test_resync_releases_an_inference_whose_trainer_event_was_missed(self):
        # [FUNCTION DOC] An unplaced trainer pod object is mutated in place to a node (no event).
        # After the fake clock advances 60 seconds (one resync) the inference pod is patched.
        """The backstop for a lost event: the next resync reconciles every key from the store again."""
        harness = PairingHarness()
        trainer = _pod(TRAINER_POOL_ID, 0, gated=False)

        async with harness.running(_pod(INFERENCE_POOL_ID, 0), trainer):
            assert harness.core_v1.patched == []

            trainer.spec.node_name = "gpu-3"
            await harness.clock.elapse(60.0)
            await settle()

            assert harness.patched_names() == [_pod_name(INFERENCE_POOL_ID, 0)]

    async def test_a_deleted_inference_is_never_patched(self):
        # [FUNCTION DOC] The inference pod is deleted by event and then the trainer is placed: no patch.
        """Scale-down deletes gated pods, and a queued reconcile must not write to a name that is gone."""
        harness = PairingHarness()
        inference = _pod(INFERENCE_POOL_ID, 0)

        async with harness.running(inference, _pod(TRAINER_POOL_ID, 0)):
            await harness.delete(inference)
            await harness.upsert(_pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False))

            assert harness.core_v1.patched == []

    async def test_a_refused_release_does_not_hold_its_neighbour_back(self):
        # [FUNCTION DOC] Half-node layout where inference pod 0's patch is rejected with 422: the
        # initial listing still results in a patch for inference pod 1 only.
        """The refusal is logged and dropped, so the gate behind it is removed in that very same pass."""
        harness = PairingHarness(layout=_sub_node_layout(), rejects={_pod_name(INFERENCE_POOL_ID, 0): 422})

        async with harness.running(
            _pod(INFERENCE_POOL_ID, 0),
            _pod(INFERENCE_POOL_ID, 1),
            _pod(TRAINER_POOL_ID, 0, 0, node_name="gpu-3", gated=False),
        ):
            assert harness.patched_names() == [_pod_name(INFERENCE_POOL_ID, 1)]
