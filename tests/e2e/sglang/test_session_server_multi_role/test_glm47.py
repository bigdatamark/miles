# [MODULE DOC] Multi-role session-server token-in-token-out end-to-end test for GLM-4.7-Flash.
# Configuration: 2 GPUs, tensor parallel 1, speculative decoding on, default cycles, the glm45
# reasoning parser and glm47 tool-call parser, append-tool recovery (the lenient chat template
# renders a tool message without validating the preceding tool-call id, so the sentinel id
# round-trips), and the Anthropic intermediate-system expectation "required". Registered for CUDA
# 2-GPU H200 (est. 900 s) and ROCm 2-GPU MI350 nightly (est. 600 s), with metric gates on the
# session mismatch rate of assistant text for session versions v1 and v2. Run as a script: calls
# the test function.
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=900, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=600, suite="nightly-stage-c-2-gpu-mi350", labels=["sglang"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


CONFIG = ModelConfig(
    model_name="zai-org/GLM-4.7-Flash",
    reasoning_parser="glm45",
    tool_call_parser="glm47",
    tito_model="glm47",
    num_gpus=2,
    tp_size=1,
    enable_spec=True,
    # Lenient template: tool message is rendered without validating that the
    # preceding assistant carries a matching tool_call.id, so the APPEND_TOOL
    # sentinel ("tool_call_id": "none") roundtrips cleanly.
    tool_call_failure_mode="append_tool",
    anthropic_intermediate_system_expectation="required",
)


def test_glm47():
    # [FUNCTION DOC] Runs the full per-model verification matrix for this module's configuration
    # through the shared runner (OpenAI on session v1 and v2, Anthropic on v2); any failure raised
    # by the runner fails the test.
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_glm47()
