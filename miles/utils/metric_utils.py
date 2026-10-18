# [MODULE DOC]
# Metric helpers: key prefixing/namespacing per trainer model, pass@k estimation for grouped
# rewards, summary statistics, compression ratio computation and a bounded repetition detector for
# generated text (windows of 10,000 characters scanned with a stride, at most 32 windows, flagged
# when a window compresses by more than 10 times).
import math
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np


REPETITION_WINDOW_SIZE_CHARS = 10_000
REPETITION_WINDOW_STRIDE_CHARS = 5_000
# has_repetition runs synchronously on the rollout executor's event loop, so
# the scan must stay bounded however long a response grows: past ~165k chars
# the stride widens to keep at most this many windows. Adjacent windows still
# overlap (stride <= window size) up to ~320k chars, so coverage stays
# gap-free for any realistic response length.
REPETITION_MAX_WINDOWS = 32
REPETITION_COMPRESSION_RATIO_THRESHOLD = 10.0


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    # [FUNCTION DOC] New mapping with the prefix prepended to every key.
    return {f"{prefix}{k}": v for k, v in d.items()}


def namespace_metrics(
    log_dict: dict[str, Any], *, trainer_model_id: str | None, step_name: str, step: int
) -> tuple[dict[str, Any], str]:
    # [FUNCTION DOC] Prefixes all keys with the trainer model id and a slash (no prefix when the id
    # is none), adds an entry for the step under the prefixed step name holding the step value, and
    # returns the new mapping and the step key. The input is not modified.
    namespace = "" if trainer_model_id is None else f"{trainer_model_id}/"
    ans = dict_add_prefix(log_dict, namespace)
    step_key = f"{namespace}{step_name}"
    ans[step_key] = step
    return ans, step_key


def strip_metrics_namespace(log_dict: dict[str, Any], *, trainer_model_id: str | None) -> dict[str, Any]:
    # [FUNCTION DOC] Inverse of namespacing: with a none id returns the same mapping object;
    # otherwise returns a new mapping with the id-plus-slash prefix removed from keys that have it
    # (other keys unchanged).
    if trainer_model_id is None:
        return log_dict
    prefix = f"{trainer_model_id}/"
    return {key.removeprefix(prefix): value for key, value in log_dict.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    # [FUNCTION DOC] Computes pass@k for k = 1, 2, 4, ... up to the largest power of two not above the
    # group size. Returns an empty mapping when the group size is 1. The number of groups defaults to
    # the reward count divided by the group size (integer division); asserts rewards fill exactly
    # groups times group size. A reward counts as correct only when exactly equal to 1. For each k,
    # applies the unbiased estimator per group (with n = group size and the group's correct count) and
    # averages over groups. Keys are pass@k texts; values numpy floats.
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    # [FUNCTION DOC] HELPER: Array of the unbiased pass@k estimate for each problem from paired sample
    # counts and correct counts (sequences truncated to the shorter without error).
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        # [FUNCTION DOC] HELPER: One problem's estimate: 1.0 if fewer than k incorrect samples exist;
        # else one minus the product over i from n-c+1 to n of (1 - k/i), the numerically stable form
        # of one minus the ratio of binomial coefficients.
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    # [FUNCTION DOC] Mean, median, max and min of the values as plain floats. Empty input yields
    # nan means and raises a value error for max/min.
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    # [FUNCTION DOC] Compresses the data (text encoded with the given encoding, default utf-8, or
    # bytes) with the chosen algorithm (zlib default, gzip, bz2, lzma) at the given level (default 9;
    # used as the preset for lzma) and returns (original size over compressed size, percent saved).
    # Empty data returns (infinity, 0.0); a zero-length compressed output returns (infinity, 100.0).
    # Unknown algorithm raises a value error. Compression modules are imported lazily.
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def _repetition_windows(text: str) -> Iterator[str]:
    # [FUNCTION DOC] HELPER: Generator of fixed-size (10,000 character) windows. Nothing for shorter
    # texts. The stride is at least 5,000 and grows so at most 32 windows are produced (final start
    # divided by 31, rounded up). Windows start at zero and step by the stride up to the final start;
    # a last window aligned to the end of the text is added if not already produced.
    """Yield overlapping windows, including the exact final suffix."""
    if len(text) < REPETITION_WINDOW_SIZE_CHARS:
        return

    final_start = len(text) - REPETITION_WINDOW_SIZE_CHARS
    stride = max(REPETITION_WINDOW_STRIDE_CHARS, math.ceil(final_start / (REPETITION_MAX_WINDOWS - 1)))
    last_start = -1
    for start in range(0, final_start + 1, stride):
        yield text[start : start + REPETITION_WINDOW_SIZE_CHARS]
        last_start = start

    if last_start != final_start:
        yield text[final_start:]


def has_repetition(text: str) -> bool:
    # [FUNCTION DOC] True if any window's compression ratio exceeds 10 (stops at the first such
    # window). Texts shorter than one window are never flagged. Synchronous and CPU bound.
    """Return whether any overlapping window is highly compressible."""
    return any(
        compression_ratio(window)[0] > REPETITION_COMPRESSION_RATIO_THRESHOLD for window in _repetition_windows(text)
    )


def compute_rollout_step(args, rollout_id):
    # [FUNCTION DOC] Step value for logging. When the always-use-train-step flag is set, converts the
    # rollout id to an optimizer step count: rollout id times rollout batch size times samples per
    # prompt, divided (integer division) by the global batch size. Otherwise returns the rollout id.
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
