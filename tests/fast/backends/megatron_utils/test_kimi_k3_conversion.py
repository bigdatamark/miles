# [MODULE DOC]
# Unit test for the Megatron-to-HuggingFace weight conversion of the Kimi K3 model: checks that a
# fused gate/up first MLP projection is split in the right order, for both the dense MLP and the
# per-expert MoE path, including the target HuggingFace parameter names.
from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.kimi_k3 import convert_kimi_k3_to_hf


def _convert(name, shape=(4, 2)):
    # [FUNCTION DOC] HELPER: Builds an integer tensor holding 0..N-1 in row-major order with the
    # given shape (default 4 by 2), then runs the Kimi K3 converter on it with an empty
    # placeholder args object and the given Megatron parameter name. Returns the converter's list
    # of (HuggingFace name, tensor) pairs.
    param = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    return convert_kimi_k3_to_hf(SimpleNamespace(), name, param)


def test_fused_fc1_splits_gate_before_up():
    # [FUNCTION DOC] Converts a dense-layer fused fc1 weight (layer 0) and an expert fused fc1
    # weight (layer 2, expert 17, expert index suffixed directly to the name). Asserts each yields
    # two outputs in order: dense gate_proj then up_proj; expert w1 then w3 under the block-sparse
    # MoE experts path. Asserts the first half of the rows (values 0-3) is the gate and the second
    # half (values 4-7) is the up projection.
    """Megatron fuses gate and up into one linear_fc1; HF wants them separate.
    Which half is which is pure convention -- swapping them produces a model
    that runs and generates plausible-looking garbage, on both the dense MLP
    and the per-expert path.
    """
    dense = _convert("module.module.decoder.layers.0.mlp.linear_fc1.weight")
    assert [name for name, _param in dense] == [
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
    ]
    assert dense[0][1].tolist() == [[0, 1], [2, 3]]
    assert dense[1][1].tolist() == [[4, 5], [6, 7]]

    expert = _convert("module.module.decoder.layers.2.mlp.experts.linear_fc1.weight17")
    assert [name for name, _param in expert] == [
        "language_model.model.layers.2.block_sparse_moe.experts.17.w1.weight",
        "language_model.model.layers.2.block_sparse_moe.experts.17.w3.weight",
    ]
    assert expert[0][1].tolist() == [[0, 1], [2, 3]]
    assert expert[1][1].tolist() == [[4, 5], [6, 7]]
