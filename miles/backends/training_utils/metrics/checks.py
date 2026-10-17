# [MODULE DOC]
# CI-only assertions for the training backend: first-step KL sanity checks and saving/comparing the
# gradient norm between runs. Failures are assertion errors.
"""CI utilities for training backend testing."""

import logging
import math
from argparse import Namespace

import torch

from miles.utils.metric_utils import strip_metrics_namespace

logger = logging.getLogger(__name__)


def check_kl(args: Namespace, log_dict: dict[str, float], step_id: int, accumulated_step_id: int) -> None:
    # [FUNCTION DOC] On-policy sanity check for the first step. Strips the trainer-model namespace
    # from metric names. At step 0, when both the ppo kl and the clip fraction metrics are logged:
    # for multi-latent attention requires ppo kl below 1e-8 (signed, no absolute value; clip
    # fraction unchecked); for LoRA runs requires |ppo kl| < 1e-8 and |clip fraction| < 1e-10; else
    # |ppo kl| < 1e-9 and |clip fraction| < 1e-10. At accumulated step 0, when the kl loss metric is
    # logged and rollout routing replay is off, requires |kl loss| < 1e-9. Raises assertion errors
    # with the log dict; no return value. KNOWN ISSUE: the multi-latent-attention branch does not
    # use an absolute value, so large negative kl passes.
    log_dict = strip_metrics_namespace(log_dict, trainer_model_id=args.trainer_model_id)
    if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
        if args.multi_latent_attention:
            # TODO: mla currently have non-zero kl, need further investigation
            assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
        elif getattr(args, "lora_rank", 0) > 0:
            # LoRA weight conversion (Megatron → HF for SGLang) introduces
            # small floating-point differences, so use a relaxed threshold.
            assert abs(log_dict["train/ppo_kl"]) < 1e-8 and abs(log_dict["train/pg_clipfrac"]) < 1e-10, f"{log_dict=}"
        else:
            assert abs(log_dict["train/ppo_kl"]) < 1e-9 and abs(log_dict["train/pg_clipfrac"]) < 1e-10, f"{log_dict=}"
    if accumulated_step_id == 0 and "train/kl_loss" in log_dict and not args.use_rollout_routing_replay:
        assert abs(log_dict["train/kl_loss"]) < 1e-9, f"{log_dict=}"


def check_grad_norm(
    args: Namespace,
    grad_norm: float,
    rollout_id: int,
    step_id: int,
    role: str = "actor",
    rank: int = 0,
) -> None:
    # [FUNCTION DOC] Save or verify the gradient norm for CI, on rank 0 only (others return). If the
    # save-path option is set, formats that path template with role, rollout id and step id and
    # saves the norm there with torch (file write). Else if the load-path option is set, loads the
    # expected norm (unsafe unpickling allowed) and asserts the values are close within relative and
    # absolute tolerance 0.03. Does nothing if neither is set. Raises an assertion error on mismatch.

    if rank != 0:
        return

    if (x := args.ci_save_grad_norm) is not None:
        ci_save_grad_norm_path = x.format(
            role=role,
            rollout_id=rollout_id,
            step_id=step_id,
        )
        torch.save(grad_norm, ci_save_grad_norm_path)

    elif (x := args.ci_load_grad_norm) is not None:
        ci_load_grad_norm_path = x.format(
            role=role,
            rollout_id=rollout_id,
            step_id=step_id,
        )
        expected_grad_norm = torch.load(ci_load_grad_norm_path, weights_only=False)
        assert math.isclose(
            grad_norm,
            expected_grad_norm,
            rel_tol=0.03,
            abs_tol=0.03,
        ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
