# [MODULE DOC] Multi-role session-server token-in-token-out end-to-end test for the Qwen3.8 27B
# FP8 model. Configuration: 2 GPUs, default tensor parallel 1, fp8 e4m3 KV cache, mamba full
# memory ratio 4.59, qwen3 reasoning parser with the qwen3_coder tool-call parser, 2 cycles,
# append-tool recovery, assistant-text threshold 0.3 for OpenAI (observed sampling noise of 0 to
# 7 of 32 per rollout) and 1.0 for Anthropic, and the Anthropic expectation "forbidden".
# Registered for CUDA 2-GPU H200 (est. 1200 s) and ROCm 2-GPU MI350 nightly (est. 700 s), with
# metric gates on the session mismatch rate of assistant text for session versions v1 and v2. Run
# as a script: calls the test function.
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1200, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=700, suite="nightly-stage-c-2-gpu-mi350", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3.8-27B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
    tito_model="qwen38small",
    num_gpus=2,
    kv_cache_dtype="fp8_e4m3",
    mamba_full_memory_ratio=4.59,
    cycles=2,
    tool_call_failure_mode="append_tool",
    # OpenAI assistant_text drift is sampling noise here: 0-7/32 per rollout over
    # 46 nightly passes (median 2/32); 7/32 tripped the 0.2 default once.
    assistant_text_threshold=0.3,
    # Anthropic tool-call conversion changes raw assistant serialization;
    # keep this endpoint-only formatting mismatch soft while hard gates stay at 0.
    anthropic_assistant_text_threshold=1.0,
    anthropic_intermediate_system_expectation="forbidden",
)


def test_qwen38small():
    # [FUNCTION DOC] Runs the full per-model verification matrix for this module's configuration
    # through the shared runner (OpenAI on session v1 and v2, Anthropic on v2); any failure raised
    # by the runner fails the test.
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_qwen38small()
