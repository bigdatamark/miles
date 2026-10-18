# [MODULE DOC] Spec construction for Kimi K3. Fixes the model's hybrid layout (delta-attention layers
# are layer numbers 1..92 that are not multiples of 4; every fourth layer and layer 93 use latent
# attention), writes K3-specific settings onto the Megatron config, and rewrites Megatron's decoder
# block spec so each layer uses the K3 transformer layer and attention classes. Virtual pipeline
# parallelism is unsupported.
import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from .layers import KimiK3Attention, KimiK3KDAAttention, KimiK3TransformerLayer
from .ops import situ_and_mul


KIMI_K3_KDA_LAYERS = tuple(
    layer_number for layer_number in range(1, 94) if layer_number % 4 != 0 and layer_number != 93
)


def configure_kimi_k3(config) -> None:
    # [FUNCTION DOC] Mutates the config. Asserts at most 93 layers, exactly 96 attention heads and an
    # MoE latent size of 3584. Sets: delta-attention layer numbers (the module-level tuple limited to
    # the configured layer count), linear-attention head count 96, head dim 128, convolution kernel
    # size 4, gate lower bound -5.0, attention-residual block size 12, the gated activation to the
    # custom clipped activation function, MoE latent norm on, activation fusion and Transformer-Engine
    # activation off.
    assert config.num_layers <= 93
    assert config.num_attention_heads == 96
    assert config.moe_latent_size == 3584
    config.kimi_kda_layers = tuple(
        layer_number for layer_number in KIMI_K3_KDA_LAYERS if layer_number <= config.num_layers
    )
    config.kimi_linear_num_heads = 96
    config.kimi_linear_head_dim = 128
    config.kimi_linear_conv_kernel_size = 4
    config.kimi_kda_gate_lower_bound = -5.0
    config.kimi_attn_res_block_size = 12
    config.gated_activation_func = situ_and_mul
    config.moe_latent_use_norm = True
    config.bias_activation_fusion = False
    config.use_te_activation_func = False


def build_kimi_k3_spec(config, vp_stage=None):
    # [FUNCTION DOC] Asserts no virtual pipeline. Gets Megatron's decoder block spec (Transformer
    # Engine) for this stage; the block holds only this stage's layers while the MoE frequency list is
    # indexed globally, so the stage's layer offset is used. For each layer, on a deep copy: module set
    # to the K3 transformer layer; self-attention set to the delta attention if the global layer
    # number is in the delta list, else the latent attention; both layer norms set to the Transformer
    # Engine norm; for dense (non-MoE) layers, the dense MLP's first linear is set to the Transformer
    # Engine column-parallel linear (the partial's keyword submodules are mutated in place on the
    # copy). Replaces the block's layer specs and returns it.
    assert config.virtual_pipeline_model_parallel_size is None, "Kimi K3 does not support VPP yet"

    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )
    # layer_specs holds only this stage's layers; moe_layer_freq is indexed globally
    layer_offset = get_transformer_layer_offset(config, vp_stage)
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.module = KimiK3TransformerLayer
        layer_number = layer_offset + len(layer_specs) + 1
        attention = KimiK3KDAAttention if layer_number in config.kimi_kda_layers else KimiK3Attention
        layer_spec.submodules.self_attention = ModuleSpec(module=attention)
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.pre_mlp_layernorm = TENorm

        if not config.moe_layer_freq[layer_offset + len(layer_specs)]:
            # The dense MLP spec is partial(MLP.as_mlp_submodule, submodules=...).
            layer_spec.submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec


def get_kimi_k3_spec(args, config, vp_stage):
    # [FUNCTION DOC] Entry point: ignores args, configures the config for K3 (side effects), and
    # returns the block spec for the given virtual pipeline stage.
    del args
    configure_kimi_k3(config)
    return build_kimi_k3_spec(config, vp_stage=vp_stage)
