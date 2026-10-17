# [MODULE DOC]
# Configuration of the session server: an immutable, strictly validated model holding every
# launch setting the server needs (network binding, backend url, tokenizer / chat-template
# settings, replay feature flags, LoRA settings, session matching and pause modes, user hook
# paths, log-prob settings), and a function that derives it from the parsed launch args.
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SessionServerConfig(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Frozen, extra-field-forbidding configuration record (all fields are required, optional ones
    # may be none). Groups: server (host, port, instance id, backend url, request timeout);
    # tokenization (HF checkpoint, chat template path, TITO model name, chat template kwargs);
    # replay (rollout routing replay, indexer replay, sampling-support replay flags, speculative
    # algorithm, layer count and router top-k for sizing routing data); debugging (debug trajectory
    # save path); LoRA (rank, adapter path, train-only flag); session behavior (use-session-server
    # switch which may be a flag or a string, message matcher mode, pause generation mode, sample
    # picker and postprocessor import paths); sampling (top log-prob count, sampling log-prob mode,
    # rollout temperature).
    host: str
    port: int
    instance_id: str | None
    backend_url: str
    timeout: float | None
    hf_checkpoint: str | None
    chat_template_path: str | None
    tito_model: str
    apply_chat_template_kwargs: dict[str, Any] | None
    use_rollout_routing_replay: bool
    use_rollout_indexer_replay: bool
    use_sampling_support_replay: bool
    sglang_speculative_algorithm: str | None
    num_layers: int | None
    moe_router_topk: int | None
    save_debug_trajectory_data: str | None
    lora_rank: int
    lora_adapter_path: str | None
    lora_train_only: bool
    use_session_server: bool | str | None
    session_message_matcher: str
    pause_generation_mode: str | None
    session_sample_picker_path: str | None
    session_sample_postprocessor_path: str | None
    rollout_top_logprobs_num: int
    rollout_sampling_logprobs_mode: str
    rollout_temperature: float


def compute_session_server_config(
    args, *, host: str, port: int, instance_id: str | None, backend_url: str
) -> SessionServerConfig:
    # [FUNCTION DOC]
    # Build the config from the parsed launch args plus explicit host, port, instance id and
    # backend url. Mapping: the request timeout comes from the miles router timeout arg. Required
    # (attribute access raises if missing): HF checkpoint, chat template path, TITO model, chat
    # template kwargs, the three replay flags, speculative algorithm, debug trajectory path, LoRA
    # rank and adapter path, top log-prob count, sampling log-prob mode. Optional with defaults
    # (via attribute lookup): layer count none, router top-k none, LoRA train-only false, use-
    # session-server none, message matcher "strict", pause mode none, picker and postprocessor
    # paths none, rollout temperature 1.0. Validation errors from the model propagate.
    return SessionServerConfig(
        host=host,
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
        timeout=args.miles_router_timeout,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
        tito_model=args.tito_model,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        use_rollout_routing_replay=args.use_rollout_routing_replay,
        use_rollout_indexer_replay=args.use_rollout_indexer_replay,
        use_sampling_support_replay=args.use_sampling_support_replay,
        sglang_speculative_algorithm=args.sglang_speculative_algorithm,
        num_layers=getattr(args, "num_layers", None),
        moe_router_topk=getattr(args, "moe_router_topk", None),
        save_debug_trajectory_data=args.save_debug_trajectory_data,
        lora_rank=args.lora_rank,
        lora_adapter_path=args.lora_adapter_path,
        lora_train_only=getattr(args, "lora_train_only", False),
        use_session_server=getattr(args, "use_session_server", None),
        session_message_matcher=getattr(args, "session_message_matcher", "strict"),
        pause_generation_mode=getattr(args, "pause_generation_mode", None),
        session_sample_picker_path=getattr(args, "session_sample_picker_path", None),
        session_sample_postprocessor_path=getattr(args, "session_sample_postprocessor_path", None),
        rollout_top_logprobs_num=args.rollout_top_logprobs_num,
        rollout_sampling_logprobs_mode=args.rollout_sampling_logprobs_mode,
        rollout_temperature=getattr(args, "rollout_temperature", 1.0),
    )
