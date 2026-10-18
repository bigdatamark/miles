# [MODULE DOC] Tests that command-line flags parsed by the project's argument parser reach the
# Megatron transformer config: the post self-attention and post MLP layernorm flags, and the
# YaRN rope settings of the Kimi model presets (multi-latent config with yarn rope type, original
# max position 4096, beta fast 1 or 32, beta slow 1). Both tests are skipped when Megatron's
# training arguments module is unavailable. Validation hooks are stubbed out, and fused kernels
# that need GPU-only libraries are disabled on the config.
import sys

import pytest

from tests.fast.launch_scripts.model_args_harness import expand_model_args


def test_post_layernorm_flags_propagate_to_megatron(monkeypatch):
    # [FUNCTION DOC] Skips if Megatron is missing. Stubs the project's and Megatron backend's
    # argument validators (reverted by monkeypatch), sets the process arguments to a minimal
    # Megatron train backend with one layer, hidden size 8, one attention head and both
    # post-layernorm flags, parses them, asserts both flags are true, picks the params dtype from
    # the bf16/fp16 flags (bfloat16, then float16, else float32), disables the fused rope option
    # (GPU-only dependency), builds the Megatron transformer config and asserts it carries both
    # flags as true.
    pytest.importorskip("megatron.training.arguments")

    import torch
    from megatron.training.arguments import core_transformer_config_from_args

    import miles.backends.megatron_utils.arguments as megatron_arguments
    import miles.utils.arguments as miles_arguments

    monkeypatch.setattr(miles_arguments, "miles_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: None)

    argv = [
        "pytest",
        "--train-backend",
        "megatron",
        "--rollout-batch-size",
        "1",
        "--num-layers",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--post-self-attn-layernorm",
        "--post-mlp-layernorm",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = miles_arguments.parse_args()

    assert args.post_self_attn_layernorm is True
    assert args.post_mlp_layernorm is True

    if args.bf16:
        args.params_dtype = torch.bfloat16
    elif args.fp16:
        args.params_dtype = torch.float16
    else:
        args.params_dtype = torch.float32

    # apply_rope_fusion requires TransformerEngine >= 1.4, which is GPU-only
    # and not installed on CPU CI. This test only validates post-layernorm flag
    # propagation, so disable the fused kernel to avoid TransformerConfig
    # __post_init__ validation failure.
    args.apply_rope_fusion = False

    config = core_transformer_config_from_args(args)

    assert config.post_self_attn_layernorm is True
    assert config.post_mlp_layernorm is True


@pytest.mark.parametrize(
    ("model_type", "beta_fast"),
    [("kimi-k2", 1), ("kimi-k2-thinking", 1), ("kimi-k25", 32), ("kimi-k25_2layer", 32)],
)
def test_kimi_yarn_flags_propagate_to_megatron(monkeypatch, model_type, beta_fast):
    # [FUNCTION DOC] Parametrized over four Kimi presets with the expected beta-fast value. Skips
    # if Megatron is missing. Stubs the validators, sets the process arguments to the Megatron
    # train backend with rollout batch size 1 plus the preset's expanded model arguments, parses
    # them, picks the params dtype as in the other test, disables fused MoE permutation (needs
    # Transformer Engine), builds the config and asserts it is the multi-latent config type with
    # yarn rope type, original max position embeddings 4096, the expected beta fast and beta slow 1.
    pytest.importorskip("megatron.training.arguments")

    import torch
    from megatron.core.transformer.transformer_config import MLATransformerConfig
    from megatron.training.arguments import core_transformer_config_from_args

    import miles.backends.megatron_utils.arguments as megatron_arguments
    import miles.utils.arguments as miles_arguments

    monkeypatch.setattr(miles_arguments, "miles_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pytest", "--train-backend", "megatron", "--rollout-batch-size", "1", *expand_model_args(model_type)],
    )

    args = miles_arguments.parse_args()
    if args.bf16:
        args.params_dtype = torch.bfloat16
    elif args.fp16:
        args.params_dtype = torch.float16
    else:
        args.params_dtype = torch.float32

    # Fused MoE permutation requires Transformer Engine, which is not installed
    # on CPU CI and is unrelated to the YaRN configuration under test.
    args.moe_permute_fusion = False

    config = core_transformer_config_from_args(args)

    assert isinstance(config, MLATransformerConfig)
    assert config.rope_type == "yarn"
    assert config.original_max_position_embeddings == 4096
    assert config.beta_fast == beta_fast
    assert config.beta_slow == 1
