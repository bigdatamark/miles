# [MODULE DOC]
# Validation of client-supplied Tinker request content against the gateway configuration. Every
# check raises the user input error type (a client mistake that fails only that request) and
# returns nothing unless noted. Covers: LoRA model config, batch (forward / backward) payloads
# including loss function inputs and size limits, sampling payloads, sequence ids, checkpoint path
# segments, checkpoint metadata and its compatibility with a model, save options, and token ids.
import math
import re

from miles.tinker.core.types import (
    LOSS_FN_INPUTS,
    LOSS_INPUT_KEYS,
    CommandOp,
    GatewayConfig,
    ModelRecord,
    UserInputError,
)


def validate_model_config(lora_config: dict, config: GatewayConfig) -> None:
    # [FUNCTION DOC] Reject a per-model LoRA config mapping that deviates from the server-wide
    # layout. A non-null seed is rejected. For each of train attention, train MLP and train
    # unembed, a non-null requested value must equal what the gateway trains. The rank (default
    # 32) must be exactly an integer type (booleans rejected) between 1 and the gateway's maximum
    # rank inclusive.
    """Reject per-model settings that conflict with the fixed server adapter layout."""
    if lora_config.get("seed") is not None:
        raise UserInputError("lora_config.seed is not supported: adapter initialization is not per-model seedable")
    layout = {
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }
    for field, layout_trains in layout.items():
        requested = lora_config.get(field)
        if requested is not None and requested != layout_trains:
            raise UserInputError(
                f"lora_config.{field}={requested} conflicts with this gateway's adapter layout "
                f"({field}={layout_trains}); the layout is fixed by the gateway target selection at server start"
            )
    rank = lora_config.get("rank", 32)
    if type(rank) is not int or not 1 <= rank <= config.max_lora_rank:
        raise UserInputError(f"lora_config.rank must be an integer in [1, {config.max_lora_rank}], got {rank!r}")


def validate_batch_payload(op: CommandOp, payload: dict, config: GatewayConfig) -> None:
    # [FUNCTION DOC]
    # Validate a forward or forward-backward payload; non-batch ops pass without checks. The
    # payload has a list of datums (internal form: tokens, target tokens, target length, optional
    # loss inputs) and a loss function name with optional config. Checks in order: datums
    # non-empty and not above the per-request datum limit; loss function known (its required loss
    # inputs come from the table); loss function config, if given, must be an object, and for the
    # ppo and cispo functions (clip thresholds) or dro (beta) any present key must be a finite
    # number of exact int or float type; then per datum: token ids and target token ids in vocab
    # range, token count within the per-datum limit, running token total; each required loss input
    # present and of length equal to the datum's target length; no extra loss inputs that the
    # function does not read. Finally the total tokens must not exceed the per-request limit.
    if not op.is_batch():
        return
    datums = payload["datums"]
    if not datums:
        raise UserInputError("forward_backward with no data")
    if len(datums) > config.max_datums_per_request:
        raise UserInputError(f"{len(datums)} datums exceeds max_datums_per_request={config.max_datums_per_request}")
    required_inputs = LOSS_FN_INPUTS.get(payload["loss_fn"])
    if required_inputs is None:
        raise UserInputError(f"unknown loss_fn {payload['loss_fn']!r}; known: {sorted(LOSS_FN_INPUTS)}")
    loss_fn_config = payload.get("loss_fn_config")
    if loss_fn_config is not None:
        if not isinstance(loss_fn_config, dict):
            raise UserInputError("loss_fn_config must be an object")
        config_keys = {
            "ppo": ("clip_low_threshold", "clip_high_threshold"),
            "cispo": ("clip_low_threshold", "clip_high_threshold"),
            "dro": ("beta",),
        }.get(payload["loss_fn"], ())
        for key in config_keys:
            if key in loss_fn_config:
                value = loss_fn_config[key]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise UserInputError(f"loss_fn_config[{key!r}] must be a finite number")
    total_tokens = 0
    for index, datum in enumerate(datums):
        validate_token_ids(datum["tokens"], config.vocab_size, f"datum {index}: model_input")
        validate_token_ids(datum["target_tokens"], config.vocab_size, f"datum {index}: target_tokens")
        if len(datum["tokens"]) > config.max_tokens_per_datum:
            raise UserInputError(f"datum {index}: {len(datum['tokens'])} tokens exceeds {config.max_tokens_per_datum}")
        total_tokens += len(datum["tokens"])
        for wire_key in required_inputs:
            values = datum.get(LOSS_INPUT_KEYS[wire_key])
            if values is None:
                raise UserInputError(
                    f"datum {index}: loss_fn {payload['loss_fn']!r} needs loss_fn_inputs[{wire_key!r}]"
                )
            if len(values) != datum["target_len"]:
                raise UserInputError(
                    f"datum {index}: loss_fn_inputs[{wire_key!r}] has {len(values)} values "
                    f"for {datum['target_len']} target tokens"
                )
        unread = [
            wire_key
            for wire_key, datum_key in LOSS_INPUT_KEYS.items()
            if wire_key not in required_inputs and datum_key in datum
        ]
        if unread:
            raise UserInputError(
                f"datum {index}: loss_fn {payload['loss_fn']!r} does not read loss_fn_inputs {unread}"
            )
    if total_tokens > config.max_tokens_per_request:
        raise UserInputError(f"{total_tokens} tokens exceeds max_tokens_per_request={config.max_tokens_per_request}")


def validate_sample_payload(payload: dict, config: GatewayConfig) -> None:
    # [FUNCTION DOC] Validate a sampling payload: prompt token ids within vocabulary range; the
    # number of samples (default 1) an exact integer from 1 to the per-request maximum; the top-k
    # prompt log-prob count (default 0) an exact nonnegative integer.
    validate_token_ids(payload["prompt_tokens"], config.vocab_size, "prompt")
    num_samples = payload.get("num_samples", 1)
    if type(num_samples) is not int or not 1 <= num_samples <= config.max_samples_per_request:
        raise UserInputError(f"num_samples must be an integer in [1, {config.max_samples_per_request}]")
    topk = payload.get("topk_prompt_logprobs", 0)
    if type(topk) is not int or topk < 0:
        raise UserInputError("topk_prompt_logprobs must be a nonnegative integer")


def validate_seq_id(value, name: str, minimum: int = 1) -> int:
    # [FUNCTION DOC] Return the value if it is an integer (booleans pass since they are integers)
    # not below the minimum (default 1); otherwise raise a user input error naming the field.
    # request seq_ids are 1-based (last_enqueued_seq_id starts at 0); idempotency keys are 0-based
    if not isinstance(value, int) or value < minimum:
        raise UserInputError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def validate_checkpoint_segment(segment: str) -> None:
    # [FUNCTION DOC] Accept only segments that start with a letter or digit followed by up to 127
    # letters, digits, dots, underscores or hyphens (whole-string match), which excludes empty
    # text, slashes, and dot-only names such as parent directory references.
    """Reject client path segments that could escape the checkpoint root."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", segment) is None:
        raise UserInputError(f"invalid checkpoint path segment {segment!r}")


def validate_checkpoint_compatibility(meta: dict, record: ModelRecord, config: GatewayConfig, shown_path: str) -> None:
    # [FUNCTION DOC] Compare checkpoint metadata with the model record and gateway layout on:
    # base model, LoRA rank, LoRA alpha, and the three train flags. The first difference raises a
    # user input error naming the key and both values. Assumes the metadata already passed the
    # metadata validation (missing keys would raise a key error).
    """Reject settings that would change the saved tensors' meaning."""
    expected = {
        "base_model": record.base_model,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }
    for key, value in expected.items():
        if meta[key] != value:
            raise UserInputError(
                f"checkpoint {shown_path!r} was saved with {key}={meta[key]!r}; this model expects {key}={value!r}"
            )


def validate_save_options(payload: dict) -> None:
    # [FUNCTION DOC] Reject save requests that set the time to live or the user metadata fields
    # (non-null), since this gateway supports neither.
    if payload.get("ttl_seconds") is not None:
        raise UserInputError("ttl_seconds is not supported: checkpoints on this gateway do not expire")
    if payload.get("user_metadata") is not None:
        raise UserInputError("user_metadata is not supported by this gateway")


def validate_token_ids(tokens: list, vocab_size: int, field: str) -> None:
    # [FUNCTION DOC] Require every token to be exactly an integer type (booleans rejected) in the
    # range from 0 up to but excluding the vocabulary size; the error message names the field.
    if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise UserInputError(f"{field} must contain integer token IDs in [0, {vocab_size})")


def validate_checkpoint_metadata(meta, shown_path: str) -> None:
    # [FUNCTION DOC] Check the shape of loaded checkpoint metadata: it must be a mapping whose
    # entries have exactly these types: tenant digest (string), base model (string), LoRA rank
    # (integer), LoRA alpha (integer or float), and three train flags (booleans); otherwise raise a
    # user input error saying the metadata is invalid or unsupported. Also the rank must be
    # positive and alpha finite.
    fields = {
        "tenant_digest": (str,),
        "base_model": (str,),
        "lora_rank": (int,),
        "lora_alpha": (int, float),
        "train_attn": (bool,),
        "train_mlp": (bool,),
        "train_unembed": (bool,),
    }
    if not isinstance(meta, dict) or any(type(meta.get(key)) not in types for key, types in fields.items()):
        raise UserInputError(f"checkpoint {shown_path!r} has invalid or unsupported metadata")
    if meta["lora_rank"] <= 0 or not math.isfinite(meta["lora_alpha"]):
        raise UserInputError(f"checkpoint {shown_path!r} has invalid LoRA metadata")
