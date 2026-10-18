# [MODULE DOC] Multi-role session-server token-in-token-out end-to-end test for NVIDIA
# Nemotron-3 Nano 30B-A3B (FP8). Configuration: 2 GPUs reserved by the suite but the model fits one,
# so tensor parallel 1 without speculative decoding (the model has no multi-token-prediction
# head); nemotron_3 reasoning parser with the qwen3_coder tool-call parser; 2 cycles;
# assistant-text mismatch threshold 1.0 because the reasoning parser keeps a trailing newline
# (a known upstream drift); append-tool recovery; Anthropic expectation "required". Registered for
# CUDA 2-GPU H200 only (est. 1050 s), with metric gates on the session mismatch rate of assistant
# text for session versions v1 and v2. Run as a script: calls the test function.
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_both_versions

register_cuda_ci(est_time=1050, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper", "blackwell"])
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v1/assistant_text")
register_ci_gate(metric_key="rollout/tito_session_mismatch_rate/v2/assistant_text")


# This suite reserves two H200s, but Nano fits one GPU and has no MTP head, so
# use TP1 without EAGLE. The pinned SGLang serves Nano through `nemotron_3`
# with `qwen3_coder`; its trailing-newline drift still requires threshold 1.0.
CONFIG = ModelConfig(
    model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    reasoning_parser="nemotron_3",
    tool_call_parser="qwen3_coder",
    tito_model="nemotron3",
    num_gpus=2,
    tp_size=1,
    cycles=2,
    assistant_text_threshold=1.0,
    tool_call_failure_mode="append_tool",
    anthropic_intermediate_system_expectation="required",
)


def test_nemotron3():
    # [FUNCTION DOC] Runs the full per-model verification matrix for this module's configuration
    # through the shared runner (OpenAI on session v1 and v2, Anthropic on v2); any failure raised
    # by the runner fails the test.
    run_both_versions(CONFIG)


if __name__ == "__main__":
    test_nemotron3()
