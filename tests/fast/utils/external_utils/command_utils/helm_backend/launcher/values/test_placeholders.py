# [MODULE DOC] Tests for the placeholder substitution applied to rendered pod commands: the
# worker index sentinel becomes a kubelet pod index placeholder (only as a whole argument), the
# base GPU id sentinel becomes a kubelet environment variable reference only for sub-node engines
# sharing a trainer node in a colocated run (whole-node and disaggregated engines get zero), and
# invariants of the substitution table (unique sentinels and placeholders, sentinels outside the
# realistic card or rank range, no sentinel contained in another, every placeholder reaches the
# command, no sentinel survives).
import itertools

import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import LAYOUT, engine, trainer

from miles.utils.external_utils.command_utils.helm_backend.launcher.values import placeholders
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import BaseWorkerSpec


class TestTheWorkerIndex:
    # [CLASS DOC] Tests of the worker index sentinel substitution.
    def test_replaces_only_the_node_rank_with_the_kubelet_placeholder(self):
        # [FUNCTION DOC] In the default engine's command, the node rank argument becomes the
        # worker index placeholder while the base GPU id argument is the plain text 0.
        """Every pod of a group shares one command, so the rank must be the one part left to kubelet."""
        command = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--node-rank") + 1] == placeholders._WORKER_INDEX_PLACEHOLDER
        assert command[command.index("--base-gpu-id") + 1] == "0"

    def test_allows_a_command_that_never_mentions_its_rank(self):
        # [FUNCTION DOC] An engine whose launch command omits the rank builds fine, and the joined
        # command contains no worker index sentinel.
        """Some engines do not take one; what matters is that no sentinel survives into the command."""
        spec = engine().model_copy(update={"launch_command": lambda ctx: "python -m sglang.launch_server"})

        command = build_values([spec], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert str(placeholders.WORKER_INDEX_SENTINEL) not in " ".join(command)

    def test_refuses_a_spec_that_builds_the_rank_into_a_larger_argument(self):
        # [FUNCTION DOC] An engine whose command joins the node rank flag and the index with an
        # equals sign raises an assertion error saying it builds something out of its pod index.
        """Kubelet substitutes whole arguments, so --node-rank=N would reach the engine unexpanded."""
        spec = engine().model_copy(
            update={
                "launch_command": lambda ctx: f"python -m sglang.launch_server --node-rank={ctx.worker_in_cell_index}"
            }
        )

        with pytest.raises(AssertionError, match="out of its pod index"):
            build_values([spec], LAYOUT).as_values()


COLOCATE_LAYOUT = LAYOUT.model_copy(update={"colocate": True})

_LARGEST_PLAUSIBLE_CARD_OR_RANK = 1_000_000


def _engine_command(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> list[str]:
    # [FUNCTION DOC] HELPER: Builds values for the specs and plan and returns the command list
    # of the first inference engine entry.
    return build_values(specs, plan).as_values()["run"]["inferenceEngines"][0]["command"]


def _base_gpu_id_argument(command: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Returns the command element following the base GPU id flag.
    return command[command.index("--base-gpu-id") + 1]


class TestBaseGpuIdOfASubNodeEngine:
    # [CLASS DOC] Tests of the base GPU id handling, using colocated plans.
    def test_leaves_a_whole_node_engine_to_be_handed_its_cards_from_zero(self):
        # [FUNCTION DOC] An 8-GPU engine against an 8-GPU-cell trainer: base GPU id is plain 0.
        """A pod holding every card of its node is given them as devices 0..n-1, so nothing has to be told."""
        command = _engine_command(
            [engine(num_cells=2, gpus_per_engine=8), trainer(num_cells=2, gpus_per_cell=8)], COLOCATE_LAYOUT
        )

        assert _base_gpu_id_argument(command) == "0"

    def test_leaves_a_disaggregated_engine_alone_even_when_it_is_narrower_than_a_node(self):
        # [FUNCTION DOC] Two 4-GPU engine pools (offsets 0 and 16) with a 2-cell, 8-GPU trainer:
        # the pool at offset 16 lies past the trainer's GPUs and keeps base GPU id 0.
        """A pool past the trainer's gpus owns its nodes, so its cards are its own and start at zero."""
        specs = [
            engine(num_cells=1, gpus_per_engine=4, name="inference-engine-0-0", gpu_offset=0),
            engine(num_cells=1, gpus_per_engine=4, name="inference-engine-0-1", gpu_offset=16),
            trainer(num_cells=2, gpus_per_cell=8),
        ]

        entries = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]["inferenceEngines"]
        after_trainer = next(entry for entry in entries if entry["poolId"] == "inference-engine-0-1")

        assert _base_gpu_id_argument(after_trainer["command"]) == "0"

    def test_asks_the_kubelet_for_the_card_an_engine_sharing_a_trainer_node_was_given(self):
        # [FUNCTION DOC] A 4-GPU engine sharing a trainer node: the base GPU id argument is a
        # reference to the base GPU id environment variable, resolved by the kubelet.
        """Which card a shared node hands this pod is decided when the pairing seats it, not when helm renders."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert _base_gpu_id_argument(command) == "$(MILES_BASE_GPU_ID)"

    def test_substitutes_the_card_as_a_whole_argument_of_its_own(self):
        # [FUNCTION DOC] Same setup: the joined command contains no base GPU id sentinel number.
        """The kubelet expands a whole argument at a time, so a sentinel welded into one never resolves."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert str(placeholders._BASE_GPU_ID_SENTINEL) not in " ".join(command)

    def test_wraps_the_command_in_no_shell_of_its_own(self):
        # [FUNCTION DOC] Same setup: the command does not start with a bash shell wrapper.
        """The value arrives already computed, so nothing in the pod has to redo the pairing's arithmetic."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert command[:2] != ["bash", "-c"]

    def test_refuses_a_spec_that_builds_the_card_into_a_larger_argument(self):
        # [FUNCTION DOC] A sub-node engine whose command joins the base GPU id flag and the first
        # GPU id with an equals sign makes building raise an assertion error mentioning its base
        # GPU id.
        """Kubelet substitutes whole arguments, so a spelling like --gpus=N would reach sglang unexpanded."""
        spec = engine(num_cells=2, gpus_per_engine=4).model_copy(
            update={"launch_command": lambda ctx: f"python -m sglang.launch_server --base-gpu-id={ctx.gpu_ids[0]}"}
        )

        with pytest.raises(AssertionError, match="out of its base gpu id"):
            build_values([spec, trainer(num_cells=1, gpus_per_cell=8)], COLOCATE_LAYOUT).as_values()


class TestEveryPairTheTableKnows:
    # [CLASS DOC] Tests that run over the whole private substitution table, using a sub-node
    # engine sharing a trainer node in a colocated plan.
    def test_substitutes_the_rank_and_the_card_in_one_pass(self):
        # [FUNCTION DOC] The node rank argument is the worker index placeholder and the base GPU id
        # argument is the base GPU id placeholder in the same command.
        """One command carries both sentinels, and a pass that stopped at the first would ship the other raw."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert command[command.index("--node-rank") + 1] == placeholders._WORKER_INDEX_PLACEHOLDER
        assert _base_gpu_id_argument(command) == placeholders._BASE_GPU_ID_PLACEHOLDER

    def test_leaves_no_sentinel_of_any_kind_in_a_rendered_command(self):
        # [FUNCTION DOC] For every substitution in the table, the joined command contains no
        # sentinel number.
        """Every test above names one sentinel; this one refuses a table entry nobody thought to substitute."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        for substitution in placeholders._SUBSTITUTIONS:
            assert str(substitution.sentinel) not in " ".join(command)

    def test_every_placeholder_of_the_table_reached_the_command(self):
        # [FUNCTION DOC] The set of all table placeholders is a subset of the command's elements,
        # i.e. each appears as a whole argument.
        """A substitution that quietly stopped happening leaves no sentinel behind either, so absence is not enough."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert {substitution.placeholder for substitution in placeholders._SUBSTITUTIONS} <= set(command)


class TestTheSubstitutionTableItself:
    # [CLASS DOC] Static invariants of the private substitution table.
    def test_no_two_entries_share_a_sentinel(self):
        # [FUNCTION DOC] All sentinels in the table are distinct.
        """Two entries on one sentinel would make which variable a command asks for depend on table order."""
        sentinels = [substitution.sentinel for substitution in placeholders._SUBSTITUTIONS]

        assert len(set(sentinels)) == len(sentinels)

    def test_no_two_entries_share_a_placeholder(self):
        # [FUNCTION DOC] All placeholders in the table are distinct.
        """Two sentinels expanding to one variable means one of them is silently reading the other's value."""
        expansions = [substitution.placeholder for substitution in placeholders._SUBSTITUTIONS]

        assert len(set(expansions)) == len(expansions)

    def test_no_sentinel_is_a_number_a_card_or_a_rank_could_be(self):
        # [FUNCTION DOC] Every sentinel is a number larger than one million, the module's bound on
        # plausible card ids or ranks.
        """A sentinel inside the real domain would make a genuine card number expand into a kubelet variable."""
        for substitution in placeholders._SUBSTITUTIONS:
            assert substitution.sentinel > _LARGEST_PLAUSIBLE_CARD_OR_RANK

    def test_no_sentinel_reads_as_part_of_another(self):
        # [FUNCTION DOC] For every ordered pair of distinct table entries, the first sentinel's
        # text is not a substring of the second's.
        """The guard tests substring containment, so one sentinel inside another would report the wrong spec."""
        for first, second in itertools.permutations(placeholders._SUBSTITUTIONS, 2):
            assert str(first.sentinel) not in str(second.sentinel)
