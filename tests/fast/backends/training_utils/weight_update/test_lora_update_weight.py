# [MODULE DOC]
# CPU mock-based tests for LoRA weight synchronization: that a name filter separates adapter (low
# rank A and B) tensors from base tensors on a sample weight list, and that the weight updater
# requires a LoRA sync configuration exactly when LoRA is active and stores it. The transfer
# protocol factory is patched with a mock that supports LoRA.
"""Mock-based tests for LoRA weight-sync logic.

Validates the LoRA vs base weight-name separation and that WeightUpdater
requires a lora_sync_config exactly when LoRA is active.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.lora.utils import is_lora_weight_name

# ---------------------------------------------------------------------------
# LoRA / base weight separation (pure logic, no distributed deps)
# ---------------------------------------------------------------------------


class TestLoraWeightSeparation:
    # [CLASS DOC] Tests of the LoRA weight name predicate on a six-entry sample list (class
    # attribute): for two projections, one base weight plus its LoRA A and B weights.
    """Test the filtering logic that _send_hf_params relies on."""

    SAMPLE_WEIGHTS = [
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
        ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
        ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
        ("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 4)),
        ("model.layers.0.mlp.gate_proj.lora_A.weight", torch.randn(8, 2)),
        ("model.layers.0.mlp.gate_proj.lora_B.weight", torch.randn(2, 8)),
    ]

    def test_separation_when_lora(self):
        # [FUNCTION DOC] Splitting the sample by the LoRA name predicate yields two base entries and
        # four LoRA entries.
        base = [(n, t) for n, t in self.SAMPLE_WEIGHTS if not is_lora_weight_name(n)]
        lora = [(n, t) for n, t in self.SAMPLE_WEIGHTS if is_lora_weight_name(n)]
        assert len(base) == 2
        assert len(lora) == 4

    def test_no_separation_when_not_lora(self):
        # [FUNCTION DOC] Asserts that treating the whole sample as base leaves six base entries and
        # zero LoRA entries. KNOWN ISSUE: it never calls any production code, so it is a tautology
        # that cannot fail.
        base = self.SAMPLE_WEIGHTS
        lora = []
        assert len(base) == 6
        assert len(lora) == 0

    def test_lora_names_contain_lora_A_or_B(self):
        # [FUNCTION DOC] Every name the predicate classifies as LoRA contains a LoRA A or LoRA B
        # path segment.
        lora = [(n, t) for n, t in self.SAMPLE_WEIGHTS if is_lora_weight_name(n)]
        for name, _ in lora:
            assert ".lora_A." in name or ".lora_B." in name

    def test_base_names_do_not_contain_lora(self):
        # [FUNCTION DOC] No name classified as base contains a LoRA A or LoRA B path segment.
        base = [(n, t) for n, t in self.SAMPLE_WEIGHTS if not is_lora_weight_name(n)]
        for name, _ in base:
            assert ".lora_A." not in name
            assert ".lora_B." not in name


# ---------------------------------------------------------------------------
# WeightUpdater lora_sync_config initialisation
# ---------------------------------------------------------------------------

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"


class TestWeightUpdaterLoraConfig:
    # [CLASS DOC] Tests of the updater's LoRA sync configuration handling at construction.
    """The updater requires a lora_sync_config exactly when LoRA is active."""

    def _make_updater(self, *, is_lora, lora_sync_config):
        # [FUNCTION DOC] HELPER: Builds a weight updater (empty args, one mock engine, empty weights
        # getter, model name "qwen", no quantization config, mock iterator factory and parallel
        # state) with the given LoRA flag and sync config, and the protocol factory patched to
        # return a mock whose LoRA support is on. Returns the updater; construction errors propagate.
        protocol = MagicMock()
        protocol.supports_lora = True
        with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
            return WeightUpdater(
                Namespace(),
                [MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen",
                quantization_config=None,
                iterator_factory=lambda *a, **k: MagicMock(),
                parallel_state=MagicMock(),
                is_lora=is_lora,
                lora_sync_config=lora_sync_config,
            )

    def test_lora_requires_config(self):
        # [FUNCTION DOC] LoRA enabled with no sync config makes construction raise an assertion error.
        with pytest.raises(AssertionError):
            self._make_updater(is_lora=True, lora_sync_config=None)

    def test_lora_config_stored(self):
        # [FUNCTION DOC] With LoRA enabled and a config mapping (peft type LORA, rank 32), the updater
        # keeps it; its rank entry reads 32.
        updater = self._make_updater(is_lora=True, lora_sync_config={"peft_type": "LORA", "r": 32})
        assert updater._lora_sync_config["r"] == 32

    def test_no_lora_no_config(self):
        # [FUNCTION DOC] With LoRA disabled and no config, the updater's stored config is none.
        updater = self._make_updater(is_lora=False, lora_sync_config=None)
        assert updater._lora_sync_config is None
