# [MODULE DOC] Qwen3.5 / 3.6 / 3.8 hybrid model plugin for Megatron: replaces the linear-attention
# layers of the stock GPT decoder block spec with a head-sharded Gated DeltaNet self-attention
# (see linear_attn), leaving full-attention layers and MoE / MLP parts as Megatron builds them.
# Which layers are linear is read from the HuggingFace config's per-layer type list.
import copy

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import gdn_heads
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.linear_attn import GatedDeltaNet, LinearAttentionLayer, Projections


def _get_text_config(hf_config):
    # [FUNCTION DOC] HELPER: return the nested text sub-config if the HuggingFace config has one
    # (vision-language models), otherwise the config itself.
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


class Qwen3_5GatedDeltaNet(GatedDeltaNet):
    # [CLASS DOC] Gated DeltaNet with the Qwen3.5-family projection layout: four head-sharded
    # projections named as in the HuggingFace checkpoint: in_proj_qkv (group-major q/k/v rows),
    # in_proj_z (output gate), in_proj_b (beta logits) and in_proj_a (decay input).
    """Qwen3.5 / 3.6 / 3.8 GDN: ``in_proj_qkv`` (group-major rows), ``in_proj_z``, ``in_proj_b``, ``in_proj_a``."""

    def _build_projections(self):
        # [FUNCTION DOC] HELPER: create in_proj_qkv (hidden to local key groups times group q/k/v
        # width), in_proj_z (hidden to local value dim), in_proj_b and in_proj_a (each hidden to
        # local value-head count), all via the head-sharded linear factory.
        hidden, local = self.config.hidden_size, self.local
        self.in_proj_qkv = self.sharded_linear("in_proj_qkv", hidden, local.num_k_heads * local.group_qkv_dim)
        self.in_proj_z = self.sharded_linear("in_proj_z", hidden, local.value_dim)
        self.in_proj_b = self.sharded_linear("in_proj_b", hidden, local.num_v_heads)
        self.in_proj_a = self.sharded_linear("in_proj_a", hidden, local.num_v_heads)

    def project(self, x):
        # [FUNCTION DOC] Apply the four projections to x and return them as qkv, gate (z), beta
        # logits (b) and decay (a).
        return Projections(self.in_proj_qkv(x), self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x))


class Attention(LinearAttentionLayer):
    # [CLASS DOC] Self-attention module for linear-attention layers of Qwen3.5-family models. The
    # class attribute core_cls selects the GDN subclass (default Qwen3_5GatedDeltaNet) so derived
    # model plugins can swap the projection layout.
    """GDN ``self_attention``; ``core_cls`` picks the family's HF projection layout."""

    core_cls: type[GatedDeltaNet] = Qwen3_5GatedDeltaNet

    def __init__(self, args, config, layer_number: int, cp_comm_type=None, pg_collection=None, name=None):
        # [FUNCTION DOC] Build the layer from the Miles args and the Megatron config. layer_number,
        # cp_comm_type and name are accepted for the spec interface and unused. If no process-group
        # collection is given, one is built from Megatron's global groups (tensor and context
        # parallel only). Loads the HuggingFace config from args.hf_checkpoint (its text sub-config
        # for VLMs) and builds core_cls with: head geometry derived from the text config, conv
        # kernel size from the linear conv kernel dim, rms norm epsilon, TP group, the backend from
        # args.linear_attention_backend, and the gated-norm activation from the config's hidden
        # activation. The input norm is the HuggingFace Qwen3-Next RMS norm over the hidden size.
        # allgather_cp comes from args.allgather_cp.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        text_config = _get_text_config(load_hf_config(args.hf_checkpoint))
        linear_attn = self.core_cls(
            config,
            heads=gdn_heads(text_config),
            conv_kernel_size=text_config.linear_conv_kernel_dim,
            norm_eps=text_config.rms_norm_eps,
            tp_group=pg_collection.tp,
            backend=args.linear_attention_backend,
            norm_activation=text_config.hidden_act,
        )
        input_layernorm = Qwen3NextRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        super().__init__(config, linear_attn, input_layernorm, pg_collection, allgather_cp=args.allgather_cp)


def get_qwen3_5_spec(args, config, vp_stage):
    # [FUNCTION DOC] Build the Megatron transformer block spec for this pipeline stage, with
    # linear-attention layers swapped for Attention. Mutates config: when args.num_experts is falsy
    # (dense model) sets the MoE layer frequency list to all zeros so the MoE path is used
    # uniformly (comment in code says "always use the moe path for MoE models"; the effect is for
    # dense ones). Gets the stock GPT decoder block spec (Transformer Engine, with the virtual
    # pipeline stage if given). Asserts no custom pipeline layout is set. Computes this stage's layer
    # count and global layer offset. Loads the HuggingFace config (trusting remote code); if the
    # text config lacks per-layer types, derives them: every interval-th layer (config's full
    # attention interval, default 4) is full attention, the rest linear. For each layer built on
    # this stage whose type is linear attention, deep-copies that layer spec and replaces its
    # self-attention submodule with a module spec for Attention carrying args as params.
    # Returns the modified block spec.
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
