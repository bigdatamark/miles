# [MODULE DOC] Multi-role session-server token-in-token-out end-to-end test for Qwen3.5-35B-A3B
# (FP8). Configuration: tensor parallel 2 on the default 4 GPUs, speculative decoding on, qwen3
# reasoning parser with the qwen3_coder tool-call parser, 2 cycles, append-tool recovery, a soft
# Anthropic assistant-text threshold of 1.0 (the Anthropic conversion changes raw serialization;
# hard gates stay strict), and the Anthropic expectation "forbidden". Registered for CUDA 4-GPU
# H200 (est. 1400 s) and ROCm 4-GPU MI350 nightly (est. 750 s), with metric gates on the session
# mismatch rate of assistant text for session versions v1 and v2. Run as a script: calls the test
# function.
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1400, suite="stage-c-4-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=750, suite="nightly-stage-c-4-gpu-mi350", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3.5-35B-A3B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
    tito_model="qwen35",
    tp_size=2,
    enable_spec=True,
    cycles=2,
    tool_call_failure_mode="append_tool",
    # Anthropic tool-call conversion changes raw assistant serialization;
    # keep this endpoint-only formatting mismatch soft while hard gates stay at 0.
    anthropic_assistant_text_threshold=1.0,
    anthropic_intermediate_system_expectation="forbidden",
)


def test_qwen35():
    # [FUNCTION DOC] Runs the full per-model verification matrix for this module's configuration
    # through the shared runner (OpenAI on session v1 and v2, Anthropic on v2); any failure raised
    # by the runner fails the test.
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_qwen35()
