# [MODULE DOC] Test for the Hugging Face weight mapping: for a tiny one layer, two expert
# Qwen3 mixture of experts config, both the unpacked per expert checkpoint parameter names
# (gate, up and down projections for each expert) and the packed names (fused gate and up, and
# down) must resolve to the same set of packed Hugging Face model parameters.
import pytest
from transformers import Qwen3MoeConfig

from miles.utils.hf_utils.weight_mapping import HfWeightMapping


@pytest.fixture(scope="module")
def hf_mapping():
    # [FUNCTION DOC] HELPER: Module scoped fixture. Builds a tiny Qwen3 mixture of experts config
    # (vocab 16, hidden 8, intermediate 16, one layer, 2 attention and 2 key value heads, head
    # dimension 4, 2 experts, 1 expert per token, expert intermediate size 4) and returns the
    # weight mapping created from it.
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
    )
    return HfWeightMapping.from_config(config)


_PREFIX = "model.layers.0.mlp.experts"
_TARGETS = [f"{_PREFIX}.gate_up_proj", f"{_PREFIX}.down_proj"]


def _unpacked_names():
    # [FUNCTION DOC] HELPER: Returns the set of 6 per-expert weight names (experts 0 and 1, each
    # with gate, up and down projection) under the layer 0 experts prefix.
    return {
        f"{_PREFIX}.{expert}.{projection}_proj.weight" for expert in range(2) for projection in ("gate", "up", "down")
    }


def test_checkpoint_formats_resolve_to_the_same_hf_parameters(hf_mapping):
    # [FUNCTION DOC] Mapping every unpacked name through the model parameter lookup yields
    # exactly the two packed targets (fused gate-up and down); mapping the packed target names
    # themselves yields the same two targets.
    unpacked = _unpacked_names()
    assert {hf_mapping.model_parameter(name) for name in unpacked} == set(_TARGETS)
    assert {hf_mapping.model_parameter(name) for name in _TARGETS} == set(_TARGETS)
