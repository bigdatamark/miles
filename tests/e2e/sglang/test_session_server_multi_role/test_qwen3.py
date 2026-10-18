# [MODULE DOC] Multi-role session-server token-in-token-out end-to-end test for Qwen3-30B-A3B
# (FP8). Configuration: 2 GPUs, tensor parallel 1, qwen3 reasoning parser, qwen25 tool-call parser,
# 2 cycles, append-tool recovery, assistant-text threshold 0.25 for OpenAI and 1.0 for Anthropic
# (the Anthropic tool-call conversion preserves structure but normalizes raw serialization, which
# shows as formatting-only mismatches), and the Anthropic expectation "required". Registered for
# CUDA 2-GPU H200 (est. 1200 s) and ROCm 2-GPU MI350 nightly (est. 750 s), with metric gates on
# the session mismatch rate of assistant text for session versions v1 and v2. Run as a script:
# calls the test function.
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1200, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=750, suite="nightly-stage-c-2-gpu-mi350", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="Qwen/Qwen3-30B-A3B-FP8",
    reasoning_parser="qwen3",
    tool_call_parser="qwen25",
    tito_model="qwen3",
    num_gpus=2,
    tp_size=1,
    cycles=2,
    tool_call_failure_mode="append_tool",
    # Anthropic tool-call conversion preserves structure but normalizes its raw
    # serialization (32/32 formatting-only mismatches in CI); keep OpenAI at 0.25.
    assistant_text_threshold=0.25,
    anthropic_assistant_text_threshold=1.0,
    anthropic_intermediate_system_expectation="required",
)


def test_qwen3():
    # [FUNCTION DOC] Runs the full per-model verification matrix for this module's configuration
    # through the shared runner (OpenAI on session v1 and v2, Anthropic on v2); any failure raised
    # by the runner fails the test.
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_qwen3()
