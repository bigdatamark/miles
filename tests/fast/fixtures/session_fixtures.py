# [MODULE DOC]
# Fixture helper that builds a session server config with test-friendly defaults.
from __future__ import annotations

from typing import Any

from miles.rollout.session.config import SessionServerConfig


def make_session_server_config(**overrides: Any) -> SessionServerConfig:
    # [FUNCTION DOC] Returns a session server config built from defaults updated by the keyword
    # overrides (unknown names are rejected by the config constructor). Defaults: localhost, port 0, no
    # instance id, backend url on localhost port 0, timeout 30, no checkpoint or chat template, tito model
    # default, strict message matching, all replay and lora options off or empty, no speculative
    # algorithm, no pause generation mode or custom pickers, zero top logprobs in selected mode, and
    # rollout temperature 1.0.
    defaults: dict[str, Any] = dict(
        host="127.0.0.1",
        port=0,
        instance_id=None,
        backend_url="http://127.0.0.1:0",
        timeout=30,
        hf_checkpoint=None,
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        save_debug_trajectory_data=None,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        use_session_server=None,
        session_message_matcher="strict",
        pause_generation_mode=None,
        session_sample_picker_path=None,
        session_sample_postprocessor_path=None,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        rollout_temperature=1.0,
    )
    defaults.update(overrides)
    return SessionServerConfig(**defaults)
