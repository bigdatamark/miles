# [MODULE DOC]
# On-policy distillation (OPD) reward function and post-processing, used as a custom reward model
# and reward post-processor. A teacher model served over an SGLang-style HTTP generate endpoint
# scores the student's already generated tokens (prefill-only requests with log-prob return).
# Two modes, selected by the OPD log-prob top-k launch arg:
# - top-k zero: the teacher's sampled-token log-probs are stored on the sample and training
#   computes the student minus teacher log-prob difference.
# - top-k positive: per response position, a token set is chosen by a strategy (only-student,
#   only-teacher, intersection, union, xor) from the student's and teacher's top-k lists; both
#   models' log-probs are looked up on that set (extra scoring requests fill in tokens missing
#   from a model's own top-k); a weighted reverse-KL estimate per position is stored on the sample.
# Optional multi-teacher routing maps a teacher name in sample metadata to a teacher endpoint.
# Module constants: allowed strategy names and reward weight modes, derived strategy subsets that
# say which scoring data each strategy needs, a TopLogprobs alias (per position list of entries
# whose element 0 is the log-prob and element 1 the token id) and a LogprobMaps alias (per
# position mapping from token id to log-prob), and the reserved fallback teacher name "default".
import math
from argparse import Namespace
from collections.abc import Iterable
from typing import Any

import aiohttp
import torch

from miles.utils.types import Sample

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}

# Reserved teacher name in --opd-teacher-urls used as the fallback route.
DEFAULT_TEACHER_NAME = "default"


def parse_teacher_urls(values: Iterable[str] | None) -> dict[str, str]:
    # [FUNCTION DOC]
    # Turn the list of "name=url" strings into a mapping from teacher name to url (empty mapping
    # for none). Each entry is split at the first equals sign only and both parts are stripped of
    # whitespace. Raises a value error if the separator is missing or either part is empty, or if
    # a name repeats. Pure function.
    """Parse ``NAME=URL`` entries from ``--opd-teacher-urls`` into a routing map.

    Splits on the first ``=`` only, so URLs containing ``=`` (e.g. query
    strings) survive intact. Raises on malformed entries and duplicate names
    so misconfiguration fails at startup, not mid-rollout.
    """
    url_map: dict[str, str] = {}
    for value in values or []:
        name, sep, url = value.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            raise ValueError(f"Invalid --opd-teacher-urls entry {value!r}; expected NAME=URL.")
        if name in url_map:
            raise ValueError(f"Duplicate teacher name {name!r} in --opd-teacher-urls.")
        url_map[name] = url
    return url_map


def _teacher_url_for_sample(args: Namespace, sample: Sample) -> str:
    # [FUNCTION DOC] HELPER: Pick the teacher scoring url for a sample. The teacher url launch arg
    # is parsed on every call (so errors surface here). With no teacher routes configured, return
    # the reward-model url launch arg. Otherwise read the teacher name from the sample metadata
    # (only if metadata is a mapping) under the teacher-key launch arg (default "opd_teacher"):
    # a known name returns its url; an unknown or missing name returns the "default" route if
    # configured, else raises a value error naming the key (and known names when a name existed).
    """Resolve the teacher scoring endpoint for one sample.

    Without ``--opd-teacher-urls`` every sample goes to ``--rm-url`` (the
    original single-teacher path, unchanged). With it, the sample is routed by
    the teacher name in ``sample.metadata[--opd-teacher-key]``; samples whose
    name is missing or unknown fall back to the reserved ``default`` entry,
    and raise if no default is configured — silently distilling from the
    wrong teacher is worse than failing the rollout.
    """
    url_map = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
    if not url_map:
        return args.rm_url

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    key = getattr(args, "opd_teacher_key", "opd_teacher")
    name = metadata.get(key)
    if name is not None:
        url = url_map.get(str(name))
        if url is not None:
            return url
        if DEFAULT_TEACHER_NAME in url_map:
            return url_map[DEFAULT_TEACHER_NAME]
        raise ValueError(
            f"Sample metadata[{key!r}]={name!r} matches no --opd-teacher-urls name "
            f"(known: {sorted(url_map)}) and no 'default' entry is configured."
        )
    if DEFAULT_TEACHER_NAME in url_map:
        return url_map[DEFAULT_TEACHER_NAME]
    raise ValueError(f"Sample metadata is missing teacher key {key!r} and --opd-teacher-urls has no 'default' entry.")


def _get_opd_top_k(args: Namespace) -> int:
    # [FUNCTION DOC] HELPER: Return the OPD log-prob top-k launch arg as an int, treating a
    # missing or falsy value as zero and clamping negatives to zero.
    return max(0, int(getattr(args, "opd_log_prob_top_k", 0) or 0))


def _get_top_k_strategy(args: Namespace) -> str:
    # [FUNCTION DOC] HELPER: Return the top-k strategy launch arg (default only-student); raises a
    # value error if it is not one of the five known strategies.
    strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if strategy not in TOP_K_STRATEGIES:
        raise ValueError(f"Unknown OPD top-k strategy: {strategy}")
    return strategy


def _get_reward_weight_mode(args: Namespace) -> str:
    # [FUNCTION DOC] HELPER: Return the reward weight mode launch arg (default student_p); raises
    # a value error if it is not student_p, teacher_p or none.
    mode = getattr(args, "opd_reward_weight_mode", "student_p")
    if mode not in REWARD_WEIGHT_MODES:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")
    return mode


def _score_payload(
    input_ids: list[int],
    top_k: int = 0,
    token_ids: list[int] | None = None,
    token_ids_positions: list[list[int]] | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build a prefill-only scoring request body for the generate endpoint:
    # the given input token ids, greedy sampling with zero new tokens and special tokens kept,
    # input log-prob return starting from position zero. If top-k is positive, request that many
    # top log-probs per position. If per-position id lists are given (patched server feature),
    # send them as the per-position token-id log-prob request, taking precedence; else if a flat
    # token id list is non-empty, send it as the token-id log-prob request applied at every
    # position. Pure function returning a mapping.
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids_positions is not None:
        # Per-position scoring (patched sglang): one id-list per input position, so the
        # teacher returns each position's own ids (sparse) instead of the global union
        # broadcast to every position (dense O(R^2)). Aligned to logprob_start_len=0.
        payload["token_ids_logprob_positions"] = token_ids_positions
    elif token_ids:
        payload["token_ids_logprob"] = token_ids
    return payload


def _per_position_ids(top_logprobs: TopLogprobs, prompt_len: int) -> list[list[int]]:
    # [FUNCTION DOC] HELPER: Return one id list per input position: prompt_len empty lists first,
    # then for each response position the token ids of its non-null top entries (a null position
    # gives an empty list). Total length is prompt length plus response length.
    """Build one token-id list per scored input position for ``token_ids_logprob_positions``.

    ``top_logprobs`` is per response position (length == response_length). Prompt
    positions are padded with empty id-lists so the layout aligns with
    ``logprob_start_len=0`` and the existing ``_trim_input_field`` extraction
    (``values[1:][-response_length:]``) — i.e. response position r lands at index
    ``prompt_len + r``.
    """
    per_pos: list[list[int]] = [[] for _ in range(prompt_len)]
    for entries in top_logprobs:
        per_pos.append([_top_entry_token_id(e) for e in (entries or []) if e is not None])
    return per_pos


def _student_score_url(args: Namespace) -> str:
    # [FUNCTION DOC] HELPER: Url of the student's generate endpoint on the SGLang router, built
    # from the router ip and port launch args.
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


async def _post_json(url: str, payload: dict[str, Any], timeout_secs: int | float | None = None) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Network call. Open a new HTTP client session with the given total
    # timeout (none means no limit), post the payload as JSON, raise on an error status, and return
    # the decoded JSON body. A new session is created per call.
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _top_entry_token_id(entry: list[Any]) -> int:
    # [FUNCTION DOC] HELPER: Token id of a top-log-prob entry: the second element as an integer.
    return int(entry[1])


def _top_entry_logprob(entry: list[Any]) -> float:
    # [FUNCTION DOC] HELPER: Log-prob of a top-log-prob entry: the first element as a float.
    return float(entry[0])


def _top_entries_to_map(entries: Iterable[list[Any]] | None) -> dict[int, float]:
    # [FUNCTION DOC] HELPER: Convert one position's entries into a mapping from token id to
    # log-prob, skipping null entries; null or empty input gives an empty mapping. Later duplicate
    # ids overwrite earlier ones.
    if not entries:
        return {}
    return {_top_entry_token_id(entry): _top_entry_logprob(entry) for entry in entries if entry is not None}


def _trim_input_field(meta_info: dict[str, Any], field: str, response_length: int) -> list[Any]:
    # [FUNCTION DOC] HELPER: Extract the response part of a per-input-position field of the
    # scoring response metadata. Raises a value error if the field is absent or null. Drops the
    # first position (a placeholder, there is no log-prob for the first input token) and keeps the
    # last response-length entries; returns an empty list when response length is zero.
    values = meta_info.get(field)
    if values is None:
        raise ValueError(f"Teacher response is missing meta_info.{field}.")
    # SGLang's first input logprob/top-logprob position is a placeholder.
    return values[1:][-response_length:] if response_length > 0 else []


def _input_logprob_maps(response: dict[str, Any], field: str, response_length: int) -> LogprobMaps:
    # [FUNCTION DOC] HELPER: For a scoring response, trim the named metadata field to the response
    # positions and convert each position's entries into a token id to log-prob mapping. Returns
    # one mapping per response position.
    return [
        _top_entries_to_map(entries) for entries in _trim_input_field(response["meta_info"], field, response_length)
    ]


def _teacher_sampled_log_probs(response: dict[str, Any], response_length: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Return a float32 one-dimensional tensor with the teacher's log-prob
    # of each sampled response token: the first element of every trimmed input-token log-prob
    # entry. Raises a value error if the field is missing.
    input_token_logprobs = _trim_input_field(response["meta_info"], "input_token_logprobs", response_length)
    return torch.tensor([item[0] for item in input_token_logprobs], dtype=torch.float32)


def _student_top_logprobs(sample: Sample, response_length: int) -> TopLogprobs:
    # [FUNCTION DOC] HELPER: Fetch the student's per-position top log-probs recorded in the
    # sample metadata at generation time, keep the last response-length positions (none when the
    # length is zero). Raises a value error if absent, or if the resulting count differs from the
    # response length.
    top_logprobs = sample.metadata.get("opd_student_top_logprobs")
    if top_logprobs is None:
        raise ValueError(
            "Top-k OPD requires student output_top_logprobs. "
            "Ensure --opd-log-prob-top-k is set before rollout generation starts."
        )
    top_logprobs = top_logprobs[-response_length:] if response_length > 0 else []
    if len(top_logprobs) != response_length:
        raise ValueError(
            f"Student top-k logprob length mismatch: got {len(top_logprobs)}, expected {response_length}."
        )
    return top_logprobs


def _unique_ids(top_logprobs: Iterable[Iterable[list[Any]]]) -> list[int]:
    # [FUNCTION DOC] HELPER: Sorted list of distinct token ids over all positions' non-null
    # entries (null positions are skipped).
    ids = set()
    for entries in top_logprobs:
        for entry in entries or []:
            if entry is not None:
                ids.add(_top_entry_token_id(entry))
    return sorted(ids)


def _ordered_unique(ids: Iterable[int]) -> list[int]:
    # [FUNCTION DOC] HELPER: Remove duplicates keeping the first occurrence order.
    seen = set()
    ordered = []
    for token_id in ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        ordered.append(token_id)
    return ordered


def _selected_token_ids(strategy: str, student_ids: list[int], teacher_ids: list[int]) -> list[int]:
    # [FUNCTION DOC] HELPER: Choose the token ids for one position by strategy. only-student gives
    # the student ids and only-teacher the teacher ids (both as given); intersection keeps student
    # ids also in the teacher set, in student order; union is the order-preserving de-duplicated
    # student then teacher ids; xor keeps, from the student ids followed by the teacher ids, those
    # present in exactly one of the two sets (order student then teacher). Unknown strategy raises
    # a value error.
    student_set = set(student_ids)
    teacher_set = set(teacher_ids)
    if strategy == "only-student":
        return student_ids
    if strategy == "only-teacher":
        return teacher_ids
    if strategy == "intersection":
        return [token_id for token_id in student_ids if token_id in teacher_set]
    if strategy == "union":
        return _ordered_unique([*student_ids, *teacher_ids])
    if strategy == "xor":
        return [
            token_id
            for token_id in [*student_ids, *teacher_ids]
            if (token_id in student_set) != (token_id in teacher_set)
        ]
    raise ValueError(f"Unknown OPD top-k strategy: {strategy}")


def _lookup_logprob(
    token_id: int,
    primary: dict[int, float],
    fallback: dict[int, float] | None,
    *,
    source: str,
) -> float:
    # [FUNCTION DOC] HELPER: Look up a token's log-prob first in the primary mapping, then in the
    # fallback mapping (if given). Raises a value error naming the source (student or teacher) if
    # found in neither.
    if token_id in primary:
        return primary[token_id]
    if fallback is not None and token_id in fallback:
        return fallback[token_id]
    raise ValueError(f"Missing {source} logprob for token id {token_id}.")


def _reward_weights(
    student_logps: list[float],
    teacher_logps: list[float],
    mode: str,
    *,
    normalize: bool,
) -> list[float]:
    # [FUNCTION DOC] HELPER: Per-token weights for the reverse-KL sum at one position. Returns an
    # empty list when there are no tokens. The log-prob source is the student's (mode student_p),
    # the teacher's (teacher_p), or all zeros (none, giving uniform weights); unknown mode raises a
    # value error. Without normalization each weight is the plain exponential of the log-prob
    # (probability). With normalization it is a softmax over the chosen log-probs, computed with
    # max subtraction for stability, and all zeros if the denominator is zero.
    if not student_logps:
        return []
    if mode == "student_p":
        logps = student_logps
    elif mode == "teacher_p":
        logps = teacher_logps
    elif mode == "none":
        logps = [0.0] * len(student_logps)
    else:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")

    if not normalize:
        return [math.exp(logp) for logp in logps]

    max_logp = max(logps)
    exp_vals = [math.exp(logp - max_logp) for logp in logps]
    denom = sum(exp_vals)
    if denom == 0.0:
        return [0.0] * len(logps)
    return [v / denom for v in exp_vals]


def _compute_topk_reverse_kl(
    args: Namespace,
    sample: Sample,
    reward_payload: dict[str, Any],
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Compute the per-position weighted reverse-KL estimate for one sample.
    # Input: the reward payload from the reward function (teacher response, and student-on-teacher
    # response if the strategy needed it). Returns a float32 tensor of length response length
    # (empty tensor if zero).
    # Per position: build the student top map (from sample metadata) and teacher top map (from the
    # teacher response) when the strategy uses them, otherwise empty maps; teacher-on-student and
    # student-on-teacher maps come from the token-id log-prob fields of the respective responses
    # when the strategy needs them. Select token ids by strategy from the student and teacher top
    # ids. For each selected id take the student log-prob from the student top map falling back to
    # the student-on-teacher map, and the teacher log-prob from the teacher top map falling back to
    # the teacher-on-student map (missing in both raises a value error). Weights come from the
    # weight mode, normalized except for the xor strategy. The position value is the sum over
    # tokens of weight times (student log-prob minus teacher log-prob).
    response_length = sample.response_length
    if response_length == 0:
        return torch.zeros((0,), dtype=torch.float32)

    strategy = _get_top_k_strategy(args)
    weight_mode = _get_reward_weight_mode(args)

    student_top_maps = (
        [_top_entries_to_map(entries) for entries in _student_top_logprobs(sample, response_length)]
        if strategy in STUDENT_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    teacher_response = reward_payload["teacher"]
    teacher_top_maps = (
        _input_logprob_maps(teacher_response, "input_top_logprobs", response_length)
        if strategy in TEACHER_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    teacher_on_student_maps = (
        _input_logprob_maps(teacher_response, "input_token_ids_logprobs", response_length)
        if strategy in TEACHER_ON_STUDENT_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    student_on_teacher_maps = (
        _input_logprob_maps(reward_payload["student_on_teacher"], "input_token_ids_logprobs", response_length)
        if strategy in STUDENT_ON_TEACHER_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    reverse_kls = []
    normalize_weights = strategy != "xor"
    for i in range(response_length):
        student_ids = list(student_top_maps[i].keys())
        teacher_ids = list(teacher_top_maps[i].keys())
        selected_ids = _selected_token_ids(strategy, student_ids, teacher_ids)

        student_logps = []
        teacher_logps = []
        for token_id in selected_ids:
            student_logps.append(
                _lookup_logprob(
                    token_id,
                    student_top_maps[i],
                    student_on_teacher_maps[i],
                    source="student",
                )
            )
            teacher_logps.append(
                _lookup_logprob(
                    token_id,
                    teacher_top_maps[i],
                    teacher_on_student_maps[i],
                    source="teacher",
                )
            )

        weights = _reward_weights(student_logps, teacher_logps, weight_mode, normalize=normalize_weights)
        reverse_kl = sum(
            w * (s_logp - t_logp) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
        )
        reverse_kls.append(reverse_kl)

    return torch.tensor(reverse_kls, dtype=torch.float32)


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Custom reward function (async, network I/O): ask the teacher (and possibly the student) to
    # score the sample's full token sequence. Extra keyword arguments are ignored. Returns the raw
    # JSON response mapping when top-k is zero (the teacher response itself), otherwise a mapping
    # with key teacher (teacher response) and, if the strategy needs it, key student_on_teacher.
    # Steps: request timeout from the router timeout launch arg (none if unset); teacher url from
    # routing. Top-k zero: one request with just the token ids. Otherwise: the teacher request asks
    # for the teacher's own top-k (only for strategies that use teacher tops) and, for strategies
    # that need the teacher's view of the student's tokens, for the log-probs of the student's
    # top token ids, either as the union over all positions or, if the per-position launch arg is
    # on, as per-position id lists aligned to the prompt length. Then, for strategies needing the
    # student's view of the teacher's tokens, take the teacher's trimmed top log-prob entries and
    # send a second scoring request to the student router endpoint asking for those ids (union or
    # per-position). Raises value errors from the helpers (bad strategy, missing student tops) and
    # HTTP errors from failed requests.
    top_k = _get_opd_top_k(args)
    # Optional per-request timeout so a hung teacher/student scoring call cannot stall
    # the whole rollout (no-op when unset).
    request_timeout = getattr(args, "sglang_router_request_timeout_secs", None)
    # Multi-teacher routing: pick this sample's teacher endpoint (falls back to
    # --rm-url when --opd-teacher-urls is unset).
    teacher_url = _teacher_url_for_sample(args, sample)
    if top_k == 0:
        return await _post_json(teacher_url, _score_payload(sample.tokens), timeout_secs=request_timeout)

    strategy = _get_top_k_strategy(args)
    # Per-position scoring requires a patched teacher/student server that understands
    # token_ids_logprob_positions; default off so an unpatched server keeps working.
    per_position = getattr(args, "opd_topk_per_position", False)
    prompt_len = len(sample.tokens) - sample.response_length

    teacher_top_k = top_k if strategy in TEACHER_TOP_STRATEGIES else 0
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)
    else:
        student_top = None
        teacher_token_ids = None

    if student_top is not None and per_position:
        teacher_payload = _score_payload(
            sample.tokens, top_k=teacher_top_k, token_ids_positions=_per_position_ids(student_top, prompt_len)
        )
    elif teacher_token_ids is not None:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k, token_ids=teacher_token_ids)
    else:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k)
    teacher_response = await _post_json(teacher_url, teacher_payload, timeout_secs=request_timeout)

    reward_payload = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        if per_position:
            student_payload = _score_payload(
                sample.tokens, token_ids_positions=_per_position_ids(teacher_top, prompt_len)
            )
        else:
            student_payload = _score_payload(sample.tokens, token_ids=_unique_ids(teacher_top))
        reward_payload["student_on_teacher"] = await _post_json(
            _student_score_url(args), student_payload, timeout_secs=request_timeout
        )

    return reward_payload


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    # [FUNCTION DOC]
    # Reward post-processor: converts the raw teacher payloads into per-sample training signals
    # (mutating the samples) and returns two identical lists of scalar rewards, all 0.0 (pure
    # distillation, no task reward); the same list object is returned twice. Extra keyword
    # arguments are ignored. Each sample's stored reward value is the payload from the reward
    # function. With top-k positive, each sample gets its reverse-KL tensor stored. Otherwise each
    # sample gets the teacher sampled-token log-prob tensor stored, trimmed to its response length.
    """Extract OPD signals from teacher responses.

    ``--opd-log-prob-top-k=0`` preserves the original sampled-token OPD path:
    store teacher log-probs and let training compute ``student_logp - teacher_logp``.

    ``--opd-log-prob-top-k>0`` follows the practical recipe from
    "Rethinking On-Policy Distillation" by forming a top-k token set per
    response position and storing a precomputed weighted reverse-KL estimate.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    if _get_opd_top_k(args) > 0:
        for sample, reward in zip(samples, raw_rewards, strict=True):
            sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, reward)
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    teacher_log_probs = [
        _teacher_sampled_log_probs(reward, response_length)
        for reward, response_length in zip(raw_rewards, response_lengths, strict=True)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
