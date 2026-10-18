# [MODULE DOC]
# Shared test helpers for the SGLang utility tests: a lazily created temporary directory holding a
# tiny Llama-style model configuration (so engine-argument code can read a real checkpoint config
# without downloading weights), and a factory for the arguments namespace that the engine's
# server-argument computation reads. Despite living in a conftest, nothing here is a fixture.
from __future__ import annotations

import functools
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

_TINY_MODEL_CONFIG: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 256,
    "max_position_embeddings": 2048,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 1000,
}


@functools.lru_cache(maxsize=1)
def tiny_model_path() -> Path:
    # [FUNCTION DOC] HELPER: Cached (one entry, so created at most once per process). Creates a
    # fresh temporary directory with prefix "miles-tiny-model-" and writes a config file into it
    # holding the tiny model configuration above as JSON (2 layers, hidden size 128, 4 heads,
    # vocabulary 1000, bfloat16, untied embeddings). Returns the directory path. Side effects:
    # leaves the directory on disk (never removed).
    model_path = Path(tempfile.mkdtemp(prefix="miles-tiny-model-"))
    (model_path / "config.json").write_text(json.dumps(_TINY_MODEL_CONFIG))
    return model_path


def make_engine_args(**overrides: Any) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns an arguments namespace with defaults for every field the
    # server-argument computation reads: the tiny model path as HuggingFace checkpoint, seed 42,
    # no rollout offload, 8 GPUs per node, 1 GPU per engine, data/pipeline/expert parallel sizes
    # 1, routing and indexer replay off, fp16 off, LoRA rank 0 with attention q/k/v projection
    # targets, no API key or adapter path, debug flags off, multi-LoRA off, not colocated.
    # Keyword overrides replace defaults.
    """Args namespace covering every field ``_compute_server_args`` touches."""
    defaults: dict[str, Any] = dict(
        hf_checkpoint=str(tiny_model_path()),
        seed=42,
        offload_rollout=False,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_rank=0,
        lora_adapter_targets=[f"model.layers.*.self_attn.{projection}_proj" for projection in ("q", "k", "v")],
        sglang_api_key=None,
        lora_adapter_path=None,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        multi_lora=False,
        colocate=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)
