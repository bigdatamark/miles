# [MODULE DOC] HELPER: Recording, padding, merging and validating the sampler's top-k candidate token
# ids and log probs per generated token. Per-sample arrays have shape (response tokens, k), with
# unavailable slots represented by token id -1 and log prob negative infinity. Two engine modes:
# "selected" (the engine's top log prob list per token) and "support" (candidates equal the sampling
# support after filtering). Functions: configure a request, append a response's candidates to a
# sample, pad for non-trained positions, merge two turns with a gap, and validate whole samples (CPU
# heavy; intended only under CI test).
"""Record the sampler's top-k candidate log-probs at generation time."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sampling
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def configure_rollout_topk_logprobs_request(args: Namespace, request: dict[str, Any], *, openai: bool = False) -> None:
    # [FUNCTION DOC] Mutates the request in place so the engine returns candidate probabilities. No-op
    # when the launch top log prob count is zero or unset. The sampling section is the request itself
    # (OpenAI style) or its sampling parameters entry. Fills unset (none or missing) sampling settings
    # with explicit defaults so the model's generation config cannot change the distribution:
    # temperature from launch args, top_p 1.0, top_k minus one (disabled), min_p 0.0. Validates the
    # resulting sampling settings against the launch temperature and candidate count (validator may
    # raise). Then, in support mode: removes both count keys and sets the sampling log prob mode to
    # support (these REPLACE any client-supplied values: launch args win). In selected mode: removes the
    # mode key and sets the count key (top_logprobs for OpenAI style, top_logprobs_num otherwise) to k.
    """Request candidate probabilities, validating the actual per-call settings.

    The rollout args own ``top_logprobs_num`` / ``top_logprobs`` and ``sampling_logprobs_mode``
    on training requests and silently override client-supplied values.
    """
    k = args.rollout_top_logprobs_num
    if not k:
        return
    sampling = request if openai else request["sampling_params"]
    # Explicit defaults avoid model generation_config changing the distribution.
    for key, default in (("temperature", args.rollout_temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        if sampling.get(key) is None:
            sampling[key] = default
    validate_rollout_topk_logprobs_sampling(sampling, temperature=args.rollout_temperature, candidate_count=k)
    if args.rollout_sampling_logprobs_mode == "support":
        # SGLang's support mode returns the actual post-filter behavior distribution.
        request.pop("top_logprobs", None)
        request.pop("top_logprobs_num", None)
        request["sampling_logprobs_mode"] = "support"
    elif openai:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs"] = k
    else:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs_num"] = k


def append_rollout_topk_logprobs(
    sample: Sample, meta: Mapping[str, Any], k: int, *, sampling_logprobs_mode: str = "selected"
) -> None:
    # [FUNCTION DOC] Append the response's candidates to the sample's two arrays (token ids as 32-bit
    # integers, log probs as 32-bit floats; created on first use, else concatenated along the token
    # axis). Does nothing if k is zero. n = number of generated tokens in the response meta info. Source
    # field: the sampling log probs list in support mode, else the top log probs list; if n > 0 and it
    # is missing or its length differs from n, raises value error. For each token row: support mode
    # raises value error if the row has more than k entries, and takes candidate ids from the
    # sampling mask list of the meta info (key error if absent) with the row as probabilities; selected
    # mode takes the first k entries as (log prob, token id) pairs. Rows are left-aligned and the rest
    # of each row stays at the unavailable markers.
    """Append compact [response, k] arrays after appending generated tokens.

    No rescoring is allowed: with stale rollouts it would replace the behavior
    policy. Slots unavailable from the server are represented by ID -1/logp -inf.
    append_sampling_metadata validates the sampling support on the same response,
    and CI checks the whole sample with validate_rollout_topk_logprobs_sample.
    """
    if not k:
        return
    n = len(meta.get("output_token_logprobs") or [])
    support_mode = sampling_logprobs_mode == "support"
    field = "output_token_sampling_logprobs" if support_mode else "output_top_logprobs"
    rows = meta.get(field)
    if n and (rows is None or len(rows) != n):
        raise ValueError(f"Rollout top-k logprobs collection requires SGLang {field} for every generated token")
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    for i, entries in enumerate(rows or []):
        if support_mode:
            if len(entries) > k:
                raise ValueError("Rollout top-k logprobs sampling support exceeds --rollout-top-logprobs-num")
            token_ids, probabilities = meta["output_token_sampling_mask"][i], entries
        else:
            token_ids = [entry[1] for entry in entries[:k]]
            probabilities = [entry[0] for entry in entries[:k]]
        ids[i, : len(token_ids)] = token_ids
        logps[i, : len(token_ids)] = probabilities
    for name, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(sample, name)
        setattr(sample, name, values if previous is None else np.concatenate((previous, values)))


def _support_membership(ids: np.ndarray, support: RolloutSamplingMask) -> tuple[np.ndarray, np.ndarray]:
    # [FUNCTION DOC] HELPER: For candidate ids of shape (tokens, k), compute (membership, lengths):
    # membership is a boolean array of the same shape, true where the candidate id is non-negative AND
    # appears in that token's support set; lengths is the support size per token. Vectorized: combine
    # row index and id into one 64-bit key (row times 2 to the 32 plus id), sort the support keys, and
    # binary-search each candidate key. With zero tokens returns an all-false array and an empty
    # lengths array. KNOWN ISSUE: if there are tokens but the support holds no ids at all, the clamped
    # lookup index of minus one on an empty array raises an index error.
    """Match candidate IDs to ragged support IDs without a per-token Python loop."""
    n = len(ids)
    if not n:
        return np.zeros_like(ids, dtype=bool), np.empty(0, dtype=np.int64)
    flat_ids, offsets = support._as_tensors()
    flat_ids = flat_ids.numpy()
    offsets = offsets.numpy()
    lengths = np.diff(offsets[: n + 1])
    support_ids = flat_ids[: offsets[n]]
    stride = 1 << 32  # Token IDs are nonnegative int32; rows remain distinct.
    support_rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    support_keys = np.sort(support_rows * stride + support_ids.astype(np.int64))
    candidate_keys = np.arange(n, dtype=np.int64)[:, None] * stride + ids.astype(np.int64)
    positions = np.searchsorted(support_keys, candidate_keys)
    membership = (ids >= 0) & (support_keys[np.minimum(positions, len(support_keys) - 1)] == candidate_keys)
    return membership, lengths


def pad_rollout_topk_logprobs(sample: Sample, count: int) -> None:
    # [FUNCTION DOC] Append count rows of unavailable markers (id minus one, log prob negative infinity)
    # to whichever of the sample's two arrays exist, keeping dtype and width; arrays that are none are
    # skipped. Mutates the sample.
    """Pad non-trained tool/observation positions before extending the response."""
    for field, fill in (("rollout_topk_token_ids", -1), ("rollout_topk_log_probs", -np.inf)):
        values = getattr(sample, field)
        if values is not None:
            padding = np.full((count, values.shape[1]), fill, dtype=values.dtype)
            setattr(sample, field, np.concatenate((values, padding)))


def validate_rollout_topk_logprobs_sample(sample: Sample, k: int) -> None:
    # [FUNCTION DOC] Check a whole sample's candidate data, raising value error on any violation. First
    # runs the sample's own validation. Requires candidate ids, candidate log probs and sampled-token
    # log probs to exist, and the candidate width to equal k. Valid slots are ids >= 0. Checks in
    # order: no duplicate valid ids within a row; every trained token (loss mask, default all ones)
    # has at least one valid candidate; if a sampling mask exists, for trained tokens the valid slots
    # equal membership in the support and the valid count equals the support size; at slots whose id
    # equals the actually sampled token on trained positions, the candidate log prob must be close
    # (absolute and relative tolerance 1e-5) to the sampled-token log prob. CPU and memory heavy
    # (sorting the whole sample), so callers should run it only under CI test.
    """Validate candidate distributions, including custom producers and restored rollouts.

    Full-sample sorting and support matching have significant CPU and memory costs;
    training callers must enable this validation only under --ci-test.
    """
    sample.validate()
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if ids is None or logps is None or sample.rollout_log_probs is None:
        raise ValueError(
            "Rollout top-k logprobs collection requires candidate and sampled-token logprobs on every rollout"
        )
    if ids.shape[1] != k:
        raise ValueError("Rollout top-k logprobs candidates must have --rollout-top-logprobs-num columns")
    valid = ids >= 0
    sorted_ids = np.sort(ids, axis=-1)
    if ((sorted_ids[:, 1:] >= 0) & (sorted_ids[:, 1:] == sorted_ids[:, :-1])).any():
        raise ValueError("Duplicate rollout top-k logprobs candidate token IDs")
    active = np.asarray(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length, dtype=bool)
    if (active & ~valid.any(-1)).any():
        raise ValueError("Every trained token needs rollout top-k logprobs candidates")
    if sample.rollout_sampling_mask is not None:
        in_support, lengths = _support_membership(ids, sample.rollout_sampling_mask)
        if ((valid != in_support) & active[:, None]).any() or ((valid.sum(-1) != lengths) & active).any():
            raise ValueError("Rollout top-k logprobs candidates must equal the sampling support")
    sampled = np.asarray(sample.rollout_log_probs)
    tokens = np.asarray(sample.tokens[-sample.response_length :] if sample.response_length else [], dtype=np.int64)
    matches = (ids == tokens[:, None]) & valid & active[:, None]
    if not np.allclose(np.broadcast_to(sampled[:, None], logps.shape)[matches], logps[matches], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled and candidate probabilities must come from the same sampler distribution")


def merge_rollout_topk_logprobs_field(first: Sample, second: Sample, field: str, gap: int) -> np.ndarray | None:
    # [FUNCTION DOC] Concatenate one candidate array of two turns with "gap" rows of unavailable markers
    # between them (id minus one for the token id field, negative infinity for the log prob field).
    # Returns none if both are none; raises value error if only one exists or the widths differ.
    # Returns a new array; inputs are unchanged.
    a, b = getattr(first, field), getattr(second, field)
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field} for rollout top-k logprobs collection")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))
