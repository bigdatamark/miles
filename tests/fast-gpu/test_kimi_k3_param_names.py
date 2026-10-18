# [MODULE DOC] GPU test that every parameter of the Kimi-K3 multi-head latent attention layer and
# the Kimi delta attention layer is covered, with consistent naming, by both conversion tables: the
# bridge's attention mapping (Hugging Face to Megatron, used for loading) and the Megatron-to-HF
# layer-name table (used for weight updates and export). Instantiates small real layers using a
# single-process NCCL group and Megatron parallel state. Skipped without fla, mbridge or CUDA.
"""Every Megatron parameter of Kimi-K3's MLA and KDA attention layers has an HF name in both the mbridge
map (HF -> Megatron load) and the megatron_to_hf map (weight updates and export). The two layer kinds
share leaf names such as ``g_proj`` and ``o_proj``, so a rename made for one kind can silently drop the
other; conversion then fails only on a real checkpoint."""

import os

import sys

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"])

pytest.importorskip("fla")
pytest.importorskip("mbridge")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import MLATransformerConfig  # noqa: E402

from miles.backends.megatron_utils.megatron_to_hf.kimi_k3 import _LAYER_NAMES  # noqa: E402
from miles_plugins.mbridge.kimi_k3 import KimiK3Bridge  # noqa: E402
from miles_plugins.models.kimi_k3.layers import KimiK3Attention, KimiK3KDAAttention  # noqa: E402


@pytest.fixture(scope="module")
def config():
    # [FUNCTION DOC] HELPER: Module-scoped generator fixture. Setup: default the rendezvous host
    # and port environment variables (127.0.0.1, 29517) without overriding existing values, select GPU
    # 0, initialize a one-rank NCCL process group and Megatron model-parallel state with tensor
    # parallel size 1, seed the model-parallel RNG with 0, and build a small latent-attention
    # transformer config (2 layers, hidden 256, 4 heads, query/key-value low rank 64, head dims 32/16/
    # 32, bfloat16, layer norm epsilon 1e-5), extended with Kimi-specific attributes: delta attention
    # at layer 2 only, 4 linear heads of dim 64, conv kernel 4, gate lower bound -5. Yields the config;
    # teardown destroys the model-parallel state and process group.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        params_dtype=torch.bfloat16,
        bf16=True,
        layernorm_epsilon=1e-5,
    )
    config.kimi_kda_layers = (2,)
    config.kimi_linear_num_heads = 4
    config.kimi_linear_head_dim = 64
    config.kimi_linear_conv_kernel_size = 4
    config.kimi_kda_gate_lower_bound = -5.0
    yield config
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.parametrize("layer_number", [1, 2], ids=["mla", "kda"])
def test_every_attention_parameter_has_an_hf_name(config, layer_number):
    # [FUNCTION DOC] Parametrized over layer 1 (latent attention) and layer 2 (delta attention).
    # Builds the layer class selected by membership in the config's delta attention layers, collects
    # its parameter names prefixed with the self-attention scope, and asserts: the list is non-empty;
    # every name is a key of the bridge's attention mapping; every name is a key of the Megatron-to-HF
    # layer-name table; and for each name the bridge's HF names (formatted with layer number 0) equal
    # exactly the single name derived from the export table with the language-model layer prefix.
    layer_cls = KimiK3KDAAttention if layer_number in config.kimi_kda_layers else KimiK3Attention
    layer = layer_cls(config, layer_number=layer_number)
    names = [f"self_attention.{name}" for name, _ in layer.named_parameters()]
    assert names
    bridge_names = set(KimiK3Bridge._ATTENTION_MAPPING)
    assert [name for name in names if name not in bridge_names] == []
    assert [name for name in names if name not in _LAYER_NAMES] == []
    disagreeing = [
        name
        for name in names
        if [hf.format(layer_number=0) for hf in KimiK3Bridge._ATTENTION_MAPPING[name]]
        != [f"language_model.model.layers.0.{_LAYER_NAMES[name]}"]
    ]
    assert disagreeing == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
