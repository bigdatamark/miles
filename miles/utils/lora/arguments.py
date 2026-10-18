# [MODULE DOC]
# LoRA command line arguments, and startup validation that derives the final target module lists
# (huggingface-side targets and adapter-side targets) from the user's selectors and the model
# config, storing them on the parsed args. Also validates multi-LoRA constraints and sets the
# multi-LoRA flag.
import logging
from typing import Any

from miles.utils.hf_utils.config import load_hf_config
from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.hf_lora_targets import (
    LORA_TARGET_GROUPS,
    exclude_hf_lora_targets,
    expand_hf_lora_targets,
    expand_packed_hf_lora_targets,
    get_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)
from miles.utils.lora.utils import is_lora_enabled, matches_lora_target, targets_expert_leaves
from miles_plugins.models.inkling.lora import resolve_inkling_adapter_targets
from miles_plugins.models.kimi_k3.lora import resolve_kimi_k3_adapter_targets

logger = logging.getLogger(__name__)


def add_lora_arguments(parser):
    # [FUNCTION DOC] Registers LoRA options on the parser and returns it: rank (int, default 0
    # meaning disabled), alpha (int, 16), dropout (float, 0), type (standard or canonical split
    # q/k/v), target modules and exclude modules (comma-separated text, default none), adapter path,
    # sync-from-tensor, base cpu backup (colocate only), train-only (rollout stays on the frozen base
    # policy), shared-outer expert loras, multi-lora adapter slot count (0 disables), and a weight
    # equality check flag. Also sets the default of the sglang virtual experts option to true and adds
    # a negating flag for it.
    """Add LoRA-related arguments for Megatron backend."""
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="LoRA rank. Set to 0 to disable LoRA (default: 0)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha for scaling (default: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate (default: 0.0)",
    )
    parser.add_argument(
        "--lora-type",
        type=str,
        default="lora",
        choices=["lora", "canonical_lora"],
        help="LoRA variant to use: 'lora' (standard) or 'canonical_lora' (split Q/K/V) (default: lora)",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default=None,
        help="LoRA targets: omit or use 'all-linear' for the HF model defaults, or provide "
        "comma-separated groups (attn,mlp,unembed), HF targets, or a mix of both. "
        "Megatron module names are also accepted by the Bridge backend.",
    )
    parser.add_argument(
        "--exclude-modules",
        type=str,
        default=None,
        help="Modules to exclude from LoRA (comma-separated)",
    )
    parser.add_argument(
        "--lora-adapter-path",
        type=str,
        default=None,
        help="Path to load pre-trained LoRA adapter weights (default: None)",
    )
    parser.add_argument(
        "--lora-sync-from-tensor",
        action="store_true",
        default=False,
        help="Sync LoRA weights via tensor instead of file (more efficient)",
    )
    parser.add_argument(
        "--lora-base-cpu-backup",
        action="store_true",
        default=False,
        help=(
            "LoRA + colocate: keep SGLang-side CPU mirror of base weights "
            "and skip per-step base sync. Trades host RAM for faster "
            "onload/offload. Ignored unless --colocate and LoRA are both on."
        ),
    )
    parser.add_argument(
        "--lora-train-only",
        action="store_true",
        default=False,
        help=(
            "Train LoRA adapters in Megatron but keep rollout engines on the frozen "
            "base policy: SGLang LoRA serving and adapter weight sync are disabled "
            "(only the base weights are synced). For models without SGLang LoRA "
            "support (e.g. Inkling native LoRA)."
        ),
    )
    parser.add_argument(
        "--experts-shared-outer-loras",
        action="store_true",
        default=False,
        help="Enable shared-outer grouped-expert LoRA (gate_up lora_A and "
        "down lora_B shared across experts, expert_dim=1). Matches SGLang "
        "PR #21466's experts_shared_outer_loras=True serving contract.",
    )
    parser.add_argument(
        "--multi-lora-n-adapters",
        type=int,
        default=0,
        help="Maximum number of concurrent adapter slots for multi-LoRA. Set to 0 to disable multi-LoRA (default: 0)",
    )
    parser.add_argument(
        "--check-lora-weight-equal",
        action="store_true",
        default=False,
        help=(
            "Verify the megatron->sglang LoRA adapter weight-sync on the colocated "
            "(from_tensors) path: on every sync the trainer ships a per-tensor sha256 "
            "manifest of the adapter it sends, and each rollout engine hashes the "
            "tensors it received and fails the load on any mismatch/missing/extra "
            "name. The LoRA analogue of --check-weight-update-equal, which only "
            "covers base weights."
        ),
    )
    # required whenever expert projections are LoRA targets, inert otherwise
    # (sglang's own default is False)
    parser.set_defaults(sglang_lora_use_virtual_experts=True)
    parser.add_argument(
        "--no-sglang-lora-use-virtual-experts",
        dest="sglang_lora_use_virtual_experts",
        action="store_false",
        help="Serve MoE-expert LoRA through sglang's fused_moe_lora alignment path instead "
        "of the virtual-experts path.",
    )
    return parser


def validate_lora_args(args):
    # [FUNCTION DOC] Validates multi-LoRA first (sets the multi-lora flag). Returns if LoRA is not
    # enabled (rank positive or adapter path set). Otherwise asserts the megatron train backend and
    # a positive rank, loads the huggingface config from the checkpoint, and stores on args the
    # resolved huggingface targets and adapter targets. Then synchronizes shared-outer expert LoRA
    # with the sglang attribute (if the sglang attribute exists and this flag is on, forces the
    # sglang one on) and asserts both agree; logs a warning naming the expert layout when expert
    # leaves are targeted. Mutates args.
    validate_multi_lora_args(args)
    if not is_lora_enabled(args):
        return
    assert args.train_backend == "megatron", "LoRA injection is not implemented for FSDP; use --train-backend megatron"
    assert args.lora_rank > 0, "LoRA requires a positive --lora-rank, including when loading an adapter"
    hf_config = load_hf_config(args.hf_checkpoint)
    args.hf_lora_targets, args.lora_adapter_targets = _resolve_lora_targets(args, hf_config)

    # Training and serving must agree on shared-outer grouped-expert LoRA (expert_dim=1).
    if args.experts_shared_outer_loras and hasattr(args, "sglang_experts_shared_outer_loras"):
        args.sglang_experts_shared_outer_loras = True
    assert args.experts_shared_outer_loras == bool(
        getattr(args, "sglang_experts_shared_outer_loras", args.experts_shared_outer_loras)
    ), "experts_shared_outer_loras and sglang_experts_shared_outer_loras must agree"
    if targets_expert_leaves(args.hf_lora_targets):
        logger.warning(
            "MoE-expert LoRA layout: %s (--experts-shared-outer-loras).",
            "shared-outer" if args.experts_shared_outer_loras else "per-expert",
        )


def _resolve_lora_targets(args, hf_config):
    # [FUNCTION DOC] HELPER: Returns (huggingface targets, adapter targets). Steps: collect model
    # module names from a meta-device weight mapping (weight suffix removed); parse target and
    # exclude selector lists; if targets were given, expand paired gate/up projections into packed
    # ones, remember explicit (non-group, non-all-linear) targets, and use the expanded list;
    # assert no module matched by both an explicit target and an exclusion; if no targets were given
    # under multi-lora, use all groups; resolve groups to concrete targets and apply exclusions.
    # If every target and exclusion matches some model module, the targets are final (with
    # exclusions applied to the matched module list when exclusions exist). Else, in bridge mode,
    # normalizes legacy megatron selectors to huggingface names through the bridge (lazy import);
    # else expands against the static per-model layout. Adapter targets default to a copy of the
    # huggingface targets; in raw mode inkling models and kimi k3 use model-specific plugin
    # resolvers. Args selectors are not modified.
    """Return final HF and adapter targets without changing the CLI selectors in `args`."""
    hf_mapping = HfWeightMapping.from_config(hf_config)
    hf_modules = [name.removesuffix(".weight") for name in hf_mapping.parameter_names]
    targets = parse_lora_targets(args.target_modules)
    exclusions = parse_lora_targets(args.exclude_modules) or []
    explicit_targets = []
    if targets is not None:
        expanded = expand_packed_hf_lora_targets(targets, hf_modules)
        explicit_targets = [
            target for target in dict.fromkeys(targets + expanded) if target not in (*LORA_TARGET_GROUPS, "all-linear")
        ]
        targets = expanded
    if exclusions:
        conflicts = {
            module
            for module in hf_modules + explicit_targets + exclusions
            if any(matches_lora_target(module, target) for target in explicit_targets)
            and any(matches_lora_target(module, exclude) for exclude in exclusions)
        }
        assert not conflicts, f"Explicit LoRA targets overlap --exclude-modules: {sorted(conflicts)}"
    if targets is None and args.multi_lora:
        targets = list(LORA_TARGET_GROUPS)
    targets = resolve_hf_lora_targets(hf_config.to_dict(), target_modules=targets)
    targets = exclude_hf_lora_targets(targets, exclusions)

    if all(any(matches_lora_target(module, target) for module in hf_modules) for target in targets + exclusions):
        hf_targets = targets
        if exclusions:
            selected = [
                module for module in hf_modules if any(matches_lora_target(module, target) for target in targets)
            ]
            hf_targets = exclude_hf_lora_targets(selected, exclusions)
    elif args.megatron_to_hf_mode == "bridge":
        # Only legacy Megatron selectors need Bridge before trainer creation.
        from miles.backends.megatron_utils.lora.target_modules import normalize_lora_targets_to_hf

        hf_targets = normalize_lora_targets_to_hf(
            args.hf_checkpoint,
            targets,
            canonical=args.lora_type == "canonical_lora",
            exclude_modules=exclusions,
            explicit_targets=explicit_targets,
        )
    else:
        layout = get_hf_lora_targets(hf_config.to_dict())
        hf_targets = exclude_hf_lora_targets(expand_hf_lora_targets(targets, layout), exclusions)

    adapter_targets = list(hf_targets)
    if args.megatron_to_hf_mode == "raw":
        if hf_config.model_type in ("inkling_model", "inkling_mm_model", "inkling_text"):
            adapter_targets = resolve_inkling_adapter_targets(hf_config.to_dict(), hf_targets)
        elif hf_config.model_type == "kimi_k3":
            adapter_targets = resolve_kimi_k3_adapter_targets(
                hf_targets,
                canonical=args.lora_type == "canonical_lora",
                experts_shared_outer_loras=args.experts_shared_outer_loras,
            )
    return hf_targets, adapter_targets


def validate_multi_lora_args(args: Any) -> None:
    # [FUNCTION DOC] Sets the multi-lora flag on args (true when the adapter slot count attribute is
    # positive). If enabled, asserts (with explanatory messages): positive rank; megatron backend; no
    # allgather context parallel; pipeline parallel size 1; thd qkv format; no shared-outer expert
    # loras; optimizer not muon-like and equal to adam (default adam); not colocated; no
    # independent-dp or train fault tolerance component; no train offload; no witness module; a
    # single sglang tokenizer worker; no per-token loss. Finally forces bridge mode for megatron to
    # huggingface conversion. Mutates args.
    # KNOWN ISSUE: the muon check is redundant with the later requirement that the optimizer equals
    # adam, and the optimizer message accesses the attribute directly even though earlier checks use
    # a defaulted lookup.
    """Set ``args.multi_lora``, then validate the trainer-side constraints of
    the slot machinery. A no-op for normal runs."""
    args.multi_lora = getattr(args, "multi_lora_n_adapters", 0) > 0
    if not args.multi_lora:
        return

    assert args.lora_rank > 0, "--lora-rank must be set when --multi-lora-n-adapters > 0"
    assert args.train_backend == "megatron", "Multi-LoRA currently requires --train-backend megatron"
    assert not getattr(
        args, "allgather_cp", False
    ), "Multi-LoRA does not support --allgather-cp: per-adapter token spans assume the zigzag CP layout."
    # Adapter routing is only recompute-safe without pipelining; enforce at launch.
    assert getattr(args, "pipeline_model_parallel_size", 1) == 1, (
        "Multi-LoRA requires --pipeline-model-parallel-size 1: a pipelined schedule would "
        "recompute activations against a later micro-batch's adapter routing."
    )
    # Per-slot token spans assume sequence-major contiguous sample packing, which only 'thd' provides.
    assert getattr(args, "qkv_format", "thd") == "thd", (
        "Multi-LoRA requires --qkv-format thd: per-adapter token spans assume the "
        f"micro-batch packs samples contiguously, which bshd does not (got {args.qkv_format!r})."
    )
    assert not getattr(args, "experts_shared_outer_loras", False), (
        "Multi-LoRA does not support --experts-shared-outer-loras; MoE expert adapters "
        "use the per-expert layout. Drop the flag (and --sglang-experts-shared-outer-loras)."
    )
    assert "muon" not in str(getattr(args, "optimizer", "")).lower(), (
        "Multi-LoRA does not support Muon: per-adapter decoupled stepping is only "
        "implemented for Adam-family per-slot optimizers"
    )
    assert not args.colocate, "Multi-LoRA requires separate training and sampling GPUs to retain accumulated gradients"
    assert (
        not getattr(args, "indep_dp", False) and "train" not in args.ft_components
    ), "Multi-LoRA does not support independent-DP training; remove 'train' from --ft-components"
    assert not args.offload_train, (
        "Multi-LoRA retains per-adapter gradient accumulation in GPU buffers between "
        "train calls; --offload-train would destroy it. Disable offload for multi-LoRA."
    )
    assert not getattr(args, "enable_witness", False), (
        "Multi-LoRA runs without the distributed optimizer (per-slot LayerWise "
        "optimizers); the witness module assumes use_distributed_optimizer"
    )
    assert getattr(args, "sglang_tokenizer_worker_num", 1) == 1, (
        "Multi-LoRA requires --sglang-tokenizer-worker-num 1: dynamic adapter loading "
        "requires a single tokenizer-side LoRA registry."
    )
    assert not args.calculate_per_token_loss, (
        "Multi-LoRA normalizes each sample by its adapter batch "
        "(sample-mean); per-token loss normalization would make adapter batch weights "
        "depend on batch contents. Drop --calculate-per-token-loss."
    )
    assert (getattr(args, "optimizer", "adam") or "adam").lower() == "adam", (
        "Multi-LoRA requires --optimizer adam: the per-slot SlotOptimizer only "
        f"implements Adam semantics; got --optimizer {args.optimizer}"
    )
    args.megatron_to_hf_mode = "bridge"
