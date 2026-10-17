# [MODULE DOC]
# Argument handling glue between the project's command-line arguments and Megatron-LM. Re-exports
# Megatron's argument parser and validator, and adds a function that fills in or overrides Megatron
# argument defaults that depend on project settings (optimizer kind, precision, sequence length,
# tokenizer, vocabulary padding, model-family normalizations). Mutates the passed namespace.
import logging
import os

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding as _vocab_size_with_padding
from megatron.training.arguments import parse_args, validate_args

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.deepseek_v4.arguments import is_dsv4_model, normalize_dsv4_args
from miles_plugins.models.glm5.arguments import normalize_dsa_args

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # [FUNCTION DOC]
    # Mutates and returns the argument namespace. In order: raises not-implemented if the
    # true-on-policy flag is set (unsupported on this backend). Enables the distributed optimizer
    # only when the optimizer is unset or Adam (case-insensitive) and optimizer is not disabled for
    # debugging; forces it off when multi-LoRA adapters are configured. Sets bf16 as the inverse of
    # fp16. Defaults sequence length to 4096 when unset and sets max position embeddings equal to
    # it. If the environment variable for deprecated-Megatron compatibility equals "1", sets the
    # pre-mcore-0.14 distributed checkpoint save flag. Always sets the MoE router to use torch
    # matrix multiply (to keep log-prob differences small versus the TE gemm). If a rope type
    # attribute exists and is unset, picks yarn for multi-latent attention else rope. If vocab size
    # is given and padded vocab size is not, computes padded vocab size with Megatron's padding
    # rule. If neither tokenizer model nor type is set, uses the HF checkpoint path as the tokenizer
    # model with the HuggingFace tokenizer type (logged). Forces trust-remote-code on. Applies
    # DeepSeek-V4 normalization for matching models. If the DSA implementation is "megatron",
    # normalizes DSA args using the HF config loaded from the HF checkpoint; else if the context
    # parallel communication type is unset sets it to a list containing point-to-point.
    if getattr(args, "true_on_policy_mode", False):
        raise NotImplementedError(
            "--true-on-policy-mode is not supported on the megatron backend with this Megatron "
            "version; support lands in a follow-up PR. Use --train-backend fsdp for true-on-policy."
        )
    # Muon currently owns its sharding path, and Megatron's distributed optimizer
    # only supports Adam-family optimizers.
    args.use_distributed_optimizer = (args.optimizer is None or args.optimizer.lower() == "adam") and not getattr(
        args, "debug_disable_optimizer", False
    )
    # Multi-LoRA: per-slot LayerWise optimizers require plain DDP all-reduce.
    if getattr(args, "multi_lora_n_adapters", 0) > 0:
        args.use_distributed_optimizer = False
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # Notice(Jiajun): new megatron has removed this argument and use dp_reshardable instead of fully_shard
    if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1":
        args.dist_ckpt_save_pre_mcore_014 = True
    # Before 20260819, radixark/Megatron-LM pick torch gemm for router, which is fp32 x fp32 ->
    # fp32, and 20260819 convert the default to TE gemm which is bf16 x bf16 -> fp32. Result show
    # the TE one increase log prob diff so manually set back
    args.moe_router_use_torch_mm = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    args.trust_remote_code = True

    if is_dsv4_model(args):
        normalize_dsv4_args(args)

    if args.dsa_impl == "megatron":
        normalize_dsa_args(args, load_hf_config(args.hf_checkpoint))
    elif args.cp_comm_type is None:
        args.cp_comm_type = ["p2p"]

    return args
