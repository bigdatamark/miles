# [MODULE DOC] Manual end-to-end session-server test for DeepSeek-V3.2 with the TITO path. Defines
# a module-level model configuration (model name from an environment variable, default the
# DeepSeek-V3.2 hub id; deepseek-v3 reasoning parser, deepseekv32 tool-call parser and TITO
# model; 8 GPUs with tensor and expert parallel size 8; speculative decoding enabled) and one
# test that passes it to the shared multi-role runner. Needs 8 GPUs and the model weights.
import os

from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig, run_one

CONFIG = ModelConfig(
    model_name=os.environ.get("DEEPSEEK_V32_MODEL", "deepseek-ai/DeepSeek-V3.2"),
    reasoning_parser="deepseek-v3",
    tool_call_parser="deepseekv32",
    tito_model="deepseekv32",
    num_gpus=8,
    tp_size=8,
    ep_size=8,
    enable_spec=True,
)


def test_deepseek_v32_session_tito():
    # [FUNCTION DOC] Runs the shared one-model session-server scenario with the module config;
    # pass/fail is determined entirely by the runner. Also invoked directly by the script entry.
    run_one(CONFIG)


if __name__ == "__main__":
    test_deepseek_v32_session_tito()
