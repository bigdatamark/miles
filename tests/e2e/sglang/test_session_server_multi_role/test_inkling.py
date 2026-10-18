# [MODULE DOC] Multi-role session-server token-in-token-out end-to-end test for the Inkling-Small
# NVFP4 model. Configuration: tensor parallel 4 with the context length capped at 32768 (the
# official 1M-context H200 recipe needs tensor parallel 8; this short-session lane trades context
# for fitting on 4 GPUs), max response length 4096, prefill CUDA graph backend disabled, 2 cycles,
# 1 sample per prompt, append-tool recovery, and the Anthropic intermediate-system expectation
# "required". Registered for CUDA 4-GPU H200 only (est. 1800 s), with metric gates on the session
# mismatch rate of assistant text for session versions v1 and v2. Run as a script: calls the test
# function.
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1800, suite="stage-c-4-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="thinkingmachines/Inkling-Small-NVFP4",
    reasoning_parser="inkling",
    tool_call_parser="inkling",
    tito_model="inkling",
    tp_size=4,
    # The official 1M-context H200 recipe needs TP8. This short-session lane
    # caps the context to make TP4 viable for this bounded session test.
    context_length=32768,
    rollout_max_response_len=4096,
    cuda_graph_backend_prefill="disabled",
    cycles=2,
    n_samples_per_prompt=1,
    tool_call_failure_mode="append_tool",
    anthropic_intermediate_system_expectation="required",
)


def test_inkling():
    # [FUNCTION DOC] Runs the full per-model verification matrix for this module's configuration
    # through the shared runner (OpenAI on session v1 and v2, Anthropic on v2); any failure raised
    # by the runner fails the test.
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_inkling()
