# [MODULE DOC] Validation helpers for the rollout top-k logprobs feature (recording the sampler's
# candidate log-probs per generated token). Two checks: one on launch arguments (run once at
# startup) and one on the per-call sampling parameters of each generation request. Both only
# raise value errors; they have no side effects.
"""Rollout top-k logprobs: the sampler's candidate log-probs recorded for each generated token.

``--rollout-top-logprobs-num`` sets the width of ``Sample.rollout_topk_token_ids`` /
``rollout_topk_log_probs``; it records the sampling distribution and does not change it.
``arguments.py`` derives ``rollout_sampling_logprobs_mode`` from the rollout filters: ``selected`` reads
``output_top_logprobs``, the top-K of the full-vocabulary distribution (unfiltered sampling);
``support`` reads ``output_token_sampling_logprobs`` over the whole realized support (filtered
sampling). ``Sample.rollout_sampling_mask`` is the separate support record of sampling-support replay.
"""

import math
from collections.abc import Mapping
from typing import Any


def validate_rollout_topk_logprobs_args(args: Any) -> None:
    # [FUNCTION DOC] Validates the parsed launch arguments. Reads the candidate count
    # (rollout_top_logprobs_num): raises a value error if negative; returns silently if zero
    # (feature off). Otherwise raises a value error if the logprobs mode is "support" and the
    # candidate count is smaller than the rollout top-k (candidates could not hold the whole
    # sampling support), and raises a value error if on-policy-distillation is used with a
    # positive student top-k log-prob count and a top-k strategy other than "only-teacher"
    # (incompatible combination). Returns nothing.
    """Reject rollout top-k logprobs settings whose recorded candidates cannot match the sampler."""
    k = args.rollout_top_logprobs_num
    if k < 0:
        raise ValueError(f"--rollout-top-logprobs-num must be non-negative, got {k}")
    if not k:
        return
    if args.rollout_sampling_logprobs_mode == "support" and k < args.rollout_top_k:
        raise ValueError(
            "Filtered rollout sampling requires --rollout-top-logprobs-num >= --rollout-top-k "
            "so the recorded candidates can hold the whole sampling support"
        )
    opd_student_top_k = args.use_opd and args.opd_log_prob_top_k > 0 and args.opd_top_k_strategy != "only-teacher"
    if opd_student_top_k:
        raise ValueError("--rollout-top-logprobs-num cannot be combined with OPD student top-k log-probs")


def validate_rollout_topk_logprobs_sampling(
    sampling: Mapping[str, Any], *, temperature: float, candidate_count: int
) -> None:
    # [FUNCTION DOC] Validates one generation call's sampling-parameter mapping against the run's
    # rollout temperature and the candidate count. Raises a value error if: the call's
    # temperature (default: the run's) differs from the run's, or the run's temperature is not a
    # finite positive number; the call's top_k (default -1, meaning off) is greater than the
    # candidate count; min_p (default 0.0) is nonzero; any truthy constraint key among json
    # schema, regex, ebnf, structural tag, custom logit processor or logit bias is present; a
    # truthy response_format is not a mapping or has a type other than "text" (default "text");
    # tool_choice (default "auto") is something other than none/"auto"/"none"; or, unless tool
    # choice is "none", any provided tool has a truthy "strict" flag in its function section.
    # Returns nothing.
    """Require per-call sampling whose distribution the recorded candidates describe.

    Top-p/top-k bounds are checked by the rollout args and sampling-support replay;
    this adds what recorded candidates need on top of them.
    """
    if sampling.get("temperature", temperature) != temperature or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Rollout top-k logprobs collection requires the same positive rollout temperature on every generation call"
        )
    if sampling.get("top_k", -1) > candidate_count:
        raise ValueError(
            "Rollout top-k logprobs collection requires top_k <= --rollout-top-logprobs-num to hold the whole support"
        )
    if sampling.get("min_p", 0.0) != 0.0:
        raise ValueError("Rollout top-k logprobs collection requires min_p=0.0")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "custom_logit_processor", "logit_bias"):
        if sampling.get(key):
            raise ValueError(f"Rollout top-k logprobs collection does not support constrained/custom sampling ({key})")
    response_format = sampling.get("response_format")
    if response_format and (not isinstance(response_format, Mapping) or response_format.get("type", "text") != "text"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained response_format")
    tool_choice = sampling.get("tool_choice", "auto")
    if tool_choice not in (None, "auto", "none"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained tool_choice")
    if tool_choice != "none" and any(tool.get("function", {}).get("strict") for tool in sampling.get("tools") or []):
        raise ValueError("Rollout top-k logprobs collection does not support strict tool schemas")
