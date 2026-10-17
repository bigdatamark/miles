# [MODULE DOC]
# Helpers for the "sampling-support replay" feature. When replay is enabled, the rollout engine
# (SGLang) reports, for every generated token, the set of token ids that was actually eligible for
# sampling (after top-p / top-k truncation). That per-token support is stored on the sample as a
# ragged rollout sampling mask so the trainer can renormalize over the same support.
# This module (1) validates that a training request's sampling parameters are compatible with
# replay, (2) converts engine metadata into a sampling mask plus selected-token log-probs, and
# (3) keeps the mask aligned with the response when tokens are appended or turns are merged.
# Tokens that were not sampled by the model (tool/observation tokens) get a singleton support.
from argparse import Namespace
from collections.abc import Mapping, Sequence

from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def should_return_sampling_mask(
    args: Namespace,
    sampling_params: Mapping[str, object] | None = None,
    *,
    evaluation: bool = False,
) -> bool:
    # [FUNCTION DOC]
    # Decide whether a request must ask the engine to return its realized sampling support, and
    # validate the request along the way. Inputs: launch args (uses the sampling-support replay
    # flag and the rollout temperature, defaulting to 1.0 when absent), optional request sampling
    # parameters (treated as empty when none), and an evaluation flag.
    # Returns False immediately for evaluation (no validation done). Otherwise delegates to the
    # request validator and returns its boolean. Raises a value error on incompatible parameters.
    """Validate whether a training request must return its realized sampling support."""
    if evaluation:
        return False

    params = sampling_params or {}
    return validate_sampling_support_request(
        params,
        replay_enabled=args.use_sampling_support_replay,
        expected_temperature=float(getattr(args, "rollout_temperature", 1.0)),
    )


def validate_sampling_support_request(
    sampling_params: Mapping[str, object],
    *,
    replay_enabled: bool,
    expected_temperature: float | None,
) -> bool:
    # [FUNCTION DOC]
    # Validate one fully resolved training request (a mapping of sampling parameters) against the
    # replay contract. Returns True when the engine must return the sampling support (replay
    # enabled and the request is valid), False when replay is disabled and the request is
    # unbounded. Raises a value error in every violation case below.
    # Steps:
    # 1. top_p defaults to 1.0 when missing or null; it must lie in (0, 1].
    # 2. Replay disabled: top_k defaults to -1 when missing/null; if top_p < 1 or top_k > 0 the
    #    request is bounded while rollout is not, which is rejected. Otherwise return False.
    # 3. Replay enabled: top_p, top_k and temperature must all be explicitly present (not null);
    #    the error lists the missing names. top_k must be positive. A missing expected temperature
    #    is an error, and the request temperature must exactly equal the expected one.
    # 4. Logit-altering parameters must be at neutral values: frequency and presence penalty in
    #    {0, 0.0, null}; repetition penalty in {1, 1.0, null}; logit bias empty or null; custom
    #    logit processor null. Anything else is rejected because the trainer cannot reproduce it.
    # 5. Return True.
    """Validate one resolved training request against its sampling replay contract."""
    request_top_p = float(1.0 if sampling_params.get("top_p") is None else sampling_params["top_p"])
    if not 0.0 < request_top_p <= 1.0:
        raise ValueError(f"training request top_p must be in (0, 1], got {request_top_p}")

    if not replay_enabled:
        raw_top_k = sampling_params.get("top_k")
        request_top_k = -1 if raw_top_k is None else int(raw_top_k)
        if request_top_p < 1.0 or request_top_k > 0:
            raise ValueError("bounded training-request sampling requires bounded rollout sampling")
        return False

    missing_params = [name for name in ("top_p", "top_k", "temperature") if sampling_params.get(name) is None]
    if missing_params:
        raise ValueError(f"sampling-support replay requires explicit request parameters: {', '.join(missing_params)}")

    request_top_k = int(sampling_params["top_k"])
    if request_top_k <= 0:
        raise ValueError(f"training request top_k must be positive, got {request_top_k}")

    request_temperature = float(sampling_params["temperature"])
    if expected_temperature is None:
        raise ValueError("sampling-support replay requires a registered training temperature")
    if request_temperature != expected_temperature:
        raise ValueError(
            f"request temperature {request_temperature} does not match the training temperature "
            f"{expected_temperature}"
        )

    unsupported = {
        "frequency_penalty": (0, 0.0, None),
        "presence_penalty": (0, 0.0, None),
        "repetition_penalty": (1, 1.0, None),
        "logit_bias": ({}, None),
        "custom_logit_processor": (None,),
    }
    for name, allowed_values in unsupported.items():
        if sampling_params.get(name) not in allowed_values:
            raise ValueError(
                f"{name} is not supported with sampling-support replay because "
                "the trainer cannot reproduce its logit transformation"
            )
    return True


def _sampling_mask_from_supports(
    token_ids: Sequence[int],
    supports: Sequence[Sequence[int]],
) -> RolloutSamplingMask:
    # [FUNCTION DOC] HELPER: Build a ragged sampling mask from per-token support lists. Requires
    # equal lengths of tokens and supports, every support non-empty, and every sampled token to be
    # a member of its own support; otherwise raises a value error. Does not mutate inputs.
    if len(token_ids) != len(supports):
        raise ValueError(f"sampling support length {len(supports)} != token length {len(token_ids)}")

    for token_id, support in zip(token_ids, supports, strict=True):
        if not support:
            raise ValueError("sampling support must contain at least one token")
        if int(token_id) not in support:
            raise ValueError(f"sampled token {token_id} is absent from its sampling support")
    return RolloutSamplingMask.from_mask_list(supports)


def append_sampling_metadata(
    sample: Sample,
    output_token_ids: Sequence[int],
    meta_info: dict,
    *,
    aborted: bool = False,
    sampling_logprobs_mode: str = "selected",
) -> list[float]:
    # [FUNCTION DOC]
    # Read SGLang's per-token sampling mask and sampling log-probs from the response metadata,
    # append the mask to the sample (mutating it), and return one float log-prob per output token.
    # Inputs: the sample, the generated token ids, the engine meta info mapping, an aborted flag,
    # and the log-prob mode: "selected" (one scalar per token) or "support" (one row per token,
    # aligned with that token's support ids, from which the sampled token's entry is picked).
    # Missing-metadata case: if either the mask or the log-probs key is absent, and the request was
    # aborted (flag set or finish reason type is abort) with no output tokens, an empty mask is
    # appended and an empty list returned; any other missing case raises a value error asking for
    # an SGLang build with native sampling-mask return.
    # Validation (value errors): log-prob count must equal output token count; supports must pass
    # the helper checks (lengths, non-empty, contains sampled token). In support mode each row must
    # be a list/tuple of the same length as its support and the sampled token must appear exactly
    # once in the support. In selected mode no row may be a list/tuple. Unknown mode raises.
    # The mask is appended only after all validation succeeded, so failures leave the sample intact.
    """Append SGLang's realized support and return sampled-token log-probs."""
    supports = meta_info.get("output_token_sampling_mask")
    log_probs = meta_info.get("output_token_sampling_logprobs")
    if supports is None or log_probs is None:
        finish_reason = meta_info.get("finish_reason") or {}
        if (aborted or finish_reason.get("type") == "abort") and not output_token_ids:
            _append_sampling_mask(sample, RolloutSamplingMask.from_mask_list([]))
            return []
        raise ValueError(
            "SGLang response is missing output_token_sampling_mask or "
            "output_token_sampling_logprobs; use an SGLang build with the "
            "native return_sampling_mask primitive"
        )
    if len(log_probs) != len(output_token_ids):
        raise ValueError(f"sampling log-prob length {len(log_probs)} != output token length {len(output_token_ids)}")

    sampling_mask = _sampling_mask_from_supports(output_token_ids, supports)
    if sampling_logprobs_mode == "support":
        selected = []
        for token_id, support, row in zip(output_token_ids, supports, log_probs, strict=True):
            if not isinstance(row, (list, tuple)) or len(row) != len(support):
                raise ValueError("SGLang support log-probs must align with the sampling support IDs")
            matches = [i for i, candidate in enumerate(support) if candidate == token_id]
            if len(matches) != 1:
                raise ValueError("Sampled token must occur exactly once in its sampling support")
            selected.append(float(row[matches[0]]))
    elif sampling_logprobs_mode == "selected":
        if any(isinstance(row, (list, tuple)) for row in log_probs):
            raise ValueError("SGLang selected sampling log-probs must be scalar")
        selected = [float(value) for value in log_probs]
    else:
        raise ValueError(f"Unsupported sampling logprobs mode: {sampling_logprobs_mode}")
    _append_sampling_mask(sample, sampling_mask)
    return selected


def append_forced_sampling_tokens(sample: Sample, token_ids: Sequence[int]) -> None:
    # [FUNCTION DOC]
    # Append a mask whose i-th entry is the singleton support containing only the i-th given token
    # id, to the sample's rollout sampling mask (mutates the sample; alignment errors from the
    # append helper propagate). Used for tokens that were inserted rather than sampled.
    """Record singleton support for non-sampled tokens inserted by the environment."""
    sampling_mask = RolloutSamplingMask.from_mask_list([[int(token_id)] for token_id in token_ids])
    _append_sampling_mask(sample, sampling_mask)


def merge_sampling_masks(
    first: Sample,
    observation_token_ids: Sequence[int],
    second: Sample,
) -> RolloutSamplingMask | None:
    # [FUNCTION DOC]
    # Return a new mask that is the concatenation of the first sample's mask, singleton supports
    # for each observation token id, and the second sample's mask, in that order. Returns None when
    # both samples have no mask; raises a value error when exactly one of them lacks a mask.
    # Does not mutate either sample.
    """Merge two per-response ragged masks with forced observation tokens between them."""
    first_mask = first.rollout_sampling_mask
    second_mask = second.rollout_sampling_mask
    if first_mask is None or second_mask is None:
        if first_mask is None and second_mask is None:
            return None
        raise ValueError("cannot merge samples unless both turns carry a complete rollout sampling mask")

    observation_mask = RolloutSamplingMask.from_mask_list([[int(token_id)] for token_id in observation_token_ids])
    return RolloutSamplingMask.concatenate((first_mask, observation_mask, second_mask))


def _append_sampling_mask(sample: Sample, sampling_mask: RolloutSamplingMask) -> None:
    # [FUNCTION DOC] HELPER: Append a mask to the sample's rollout sampling mask in place. If the
    # sample has no mask yet, the given mask becomes it, but only when the response length is still
    # zero (otherwise raises a value error because earlier tokens would lack mask entries). If a
    # mask exists, its length must equal the current response length (else a value error) and the
    # sample's mask is replaced with the concatenation of old and new.
    if sample.rollout_sampling_mask is None:
        if sample.response_length != 0:
            raise ValueError("cannot initialize a sampling mask after response tokens have already been appended")
        sample.rollout_sampling_mask = sampling_mask
        return

    if len(sample.rollout_sampling_mask) != sample.response_length:
        raise ValueError(
            f"sampling mask length {len(sample.rollout_sampling_mask)} is not aligned with "
            f"response_length {sample.response_length} before appending"
        )
    sample.rollout_sampling_mask = RolloutSamplingMask.concatenate((sample.rollout_sampling_mask, sampling_mask))
