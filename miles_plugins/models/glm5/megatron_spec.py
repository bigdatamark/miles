# [MODULE DOC] Spec provider that builds the decoder block spec with Megatron's own experimental
# deepseek-sparse-attention implementation, after aligning the top-k tie-breaking policy with the
# in-repo indexer when the flashinfer backend is selected.
"""Native Megatron DSA spec provider for the raw model-provider path."""

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

from miles.kernels.attention.dsa.topk import flashinfer_tie_break_value


def get_dsa_spec(args, config, vp_stage):
    # [FUNCTION DOC] If the config's indexer top-k backend is flashinfer: imports the sglang
    # environment lazily (so it runs inside the training worker) and sets on the config the
    # deterministic-top-k flag from the sglang environment variable and the tie-break value from the
    # in-repo helper. Then returns Megatron's block spec for the experimental attention variant for the
    # given virtual pipeline stage. The args parameter is unused. Mutates the config.
    # Resolve inside the training worker so --train-env-vars and Ray's runtime
    # environment select exactly the same tie policy as the Miles indexer.
    if config.dsa_indexer_topk_backend == "flashinfer":
        from sglang.srt.environ import envs

        config.dsa_indexer_topk_deterministic = envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get()
        config.dsa_indexer_topk_tie_break = flashinfer_tie_break_value()
    return get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage)
