# [MODULE DOC]
# Defaults for the Hub-kernel plugin: slot name constants (gated delta rule, causal conv1d, flash
# attention varlen), the three shipped kernel specs (all from the kernels-community Hub
# organization, pinned by version number), the table of functions each slot must provide, and the
# provider function returning the shipped slot mapping. A custom mapping provider replaces the
# whole mapping but each chosen slot must still declare all required functions.
"""Default `slot -> HubKernelSpec` mapping for `--kernel-backend hub`.

A slot is a role in the FSDP forward, not a repo: swap the whole mapping with
`--kernel-mapping-path`, whose signature is `(args) -> dict[str, HubKernelSpec]`. Every repo here
is under `kernels-community`, which the Hub marks as a trusted kernel publisher.
"""

from miles.backends.fsdp_utils.plugins.hf_kernels.loader import HubKernelSpec

# GatedDeltaNet's linear-attention recurrence (Qwen3-Next / Qwen3.5 / Qwen3.6). Without the
# flash-linear-attention wheel, transformers' torch fallback swallows the injected cu_seqlens
# into **kwargs, so packed documents silently stop resetting the recurrence.
SLOT_GATED_DELTA_RULE = "gated_delta_rule"

# GatedDeltaNet's short causal convolution, same architectures. Without the causal_conv1d wheel
# the forward drops to F.silu(self.conv1d(...)), which takes no seq_idx.
SLOT_CAUSAL_CONV1D = "causal_conv1d"

# The NemotronH attention mixer's varlen path; without it the packed-document attention patch
# in models/nemotron_h.py returns the unpatched dense forward.
SLOT_FLASH_ATTN_VARLEN = "flash_attn_varlen"

FLA = HubKernelSpec(
    repo_id="kernels-community/fla",
    version=1,
    functions=("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
)

CAUSAL_CONV1D = HubKernelSpec(
    repo_id="kernels-community/causal-conv1d",
    version=1,
    functions=("causal_conv1d_fn", "causal_conv1d_update"),
)

FLASH_ATTN2 = HubKernelSpec(
    repo_id="kernels-community/flash-attn2",
    version=2,
    functions=("flash_attn_varlen_func",),
)

# Custom mappings may replace repos or omit entire slots, but a selected slot must be complete.
REQUIRED_SLOT_FUNCTIONS = {
    SLOT_GATED_DELTA_RULE: FLA.functions,
    SLOT_CAUSAL_CONV1D: CAUSAL_CONV1D.functions,
    SLOT_FLASH_ATTN_VARLEN: FLASH_ATTN2.functions,
}


def default_module_kernels(args) -> dict[str, HubKernelSpec]:
    # [FUNCTION DOC]
    # Provider for the default mapping. Returns an empty mapping when either the true-on-policy
    # flag or the deterministic flag on args is set (those modes own their numerics); otherwise
    # returns a mapping from the three slot names to the linear-attention, causal-conv and
    # flash-attention-2 specs.
    """The mapping miles ships. Empty under the bit-exact run modes, which own their numerics."""
    if args.true_on_policy_mode or args.deterministic_mode:
        return {}
    return {
        SLOT_GATED_DELTA_RULE: FLA,
        SLOT_CAUSAL_CONV1D: CAUSAL_CONV1D,
        SLOT_FLASH_ATTN_VARLEN: FLASH_ATTN2,
    }
