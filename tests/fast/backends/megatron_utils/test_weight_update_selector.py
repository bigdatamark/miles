# [MODULE DOC]
# Test for the weight-update selector of the Megatron HF weight iterator base class: whether the
# whole model ("all") or only the target model ("target") is synchronized depends on whether a
# speculative-decoding draft exists and whether the trainer trains it (multi-token-prediction
# layers). The module registers itself as a CPU test in the first CPU stage with an estimated 60
# seconds and no labels (the registration sits between the docstring and the imports).
"""The weight-update selector follows the trainer model, not the conversion mode."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import MegatronHfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


class _Iterator(MegatronHfWeightIteratorBase):
    # [CLASS DOC] HELPER: Minimal concrete subclass of the iterator base class, implementing the
    # abstract hooks as empty results so only the base class's selector logic is exercised.
    def _iter_hf_param_units(self, weights, *, materialize):
        # [FUNCTION DOC] HELPER: Returns an empty iterator regardless of input.
        return iter(())

    def _export_pp_local_lora(self, adapter):
        # [FUNCTION DOC] HELPER: Returns an empty list regardless of input.
        return []


def _make(speculative, mtp_num_layers):
    # [FUNCTION DOC] HELPER: Builds the minimal iterator: one fake model chunk whose config has
    # the given number of multi-token-prediction layers, arguments with the given speculative
    # decoding algorithm and no Q LoRA rank, a placement that gathers pipeline stages, model name
    # "qwen" and no quantization config. Returns the iterator.
    model = [SimpleNamespace(config=SimpleNamespace(mtp_num_layers=mtp_num_layers))]
    args = Namespace(sglang_speculative_algorithm=speculative, q_lora_rank=None)
    return _Iterator(
        args, model, placement=WeightUpdatePlacement(gather_pp=True), model_name="qwen", quantization_config=None
    )


@pytest.mark.parametrize(
    "speculative, mtp_num_layers, selector",
    [
        (None, None, "all"),
        (None, 1, "all"),
        ("EAGLE", 1, "all"),
        ("EAGLE", None, "target"),
        ("EAGLE", 0, "target"),
    ],
    ids=["no_draft", "no_draft_with_mtp", "trained_draft", "frozen_draft", "frozen_draft_zero_layers"],
)
def test_frozen_draft_is_excluded_from_the_sync(speculative, mtp_num_layers, selector):
    # [FUNCTION DOC] Parametrized over (speculative algorithm, trainer MTP layer count, expected
    # selector): no speculative -> all (regardless of MTP layers); EAGLE with 1 MTP layer -> all
    # (draft is trained); EAGLE with none or 0 MTP layers -> target (frozen draft excluded).
    # Asserts the iterator's weight-update selector equals the expectation.
    """A draft the trainer never feeds must not be re-postprocessed on every sync."""
    assert _make(speculative, mtp_num_layers).weight_update_selector == selector
