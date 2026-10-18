# [MODULE DOC] End-to-end pytest module (one H200) checking that session server v2 reproduces the
# training payload of session server v1 bit for bit on agentic trajectories with same-prompt
# retries, using Qwen3-8B at a pinned revision (path overridable by environment variable) with
# fa3 attention, deterministic inference, radix cache disabled, qwen3 reasoning parser, qwen25
# tool-call parser and static memory fraction 0.5. Sixteen input samples (indices and rollout ids
# 0 through 15, the initial agent messages as prompt, reward 0.25, a small metadata mapping) are run
# through both versions via shared helpers from the tests session-parity utilities, then compared
# pairwise. Registered for CUDA 2-GPU H200 (est. 190 s, Hopper only), label sglang. Run as a
# script: runs pytest on this file verbosely.
# KNOWN ISSUE: registered for the 2-GPU stage although the server starts without tensor parallelism
# and the fixture only requires one CUDA device.
import os

import pytest
import torch
from huggingface_hub import snapshot_download
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils.sglang_server import start_sglang_server
from tests.session_parity_utils import V1, V2, assert_agentic_retry_trajectory_parity, run_agentic_retry_trajectories

from miles.utils.test_utils.session_verify_agent import build_initial_messages
from miles.utils.types import Sample

register_cuda_ci(est_time=190, suite="stage-c-2-gpu-h200", labels=["sglang"], hardware=["hopper"])

_MODEL_ID = "Qwen/Qwen3-8B"
_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
_DEFAULT_MODEL_PATH = "/root/models/Qwen3-8B"
_MODEL_PATH_OVERRIDE = os.environ.get("SGLANG_SESSION_PARITY_MODEL_PATH")
_MODEL_PATH = _MODEL_PATH_OVERRIDE or _DEFAULT_MODEL_PATH
_BATCH_SIZE = 16


@pytest.fixture(scope="module")
def sglang_server():
    # [FUNCTION DOC] HELPER: Module-scoped pytest fixture. Asserts CUDA is available and the first
    # GPU is an H200. Downloads the pinned model revision unless the path override is set. Starts an
    # SGLang server with deterministic inference and the extra arguments described in the module
    # doc, yields it, and always stops it at teardown.
    assert torch.cuda.is_available()
    assert "H200" in torch.cuda.get_device_name(0)
    if _MODEL_PATH_OVERRIDE is None:
        snapshot_download(_MODEL_ID, revision=_MODEL_REVISION, local_dir=_DEFAULT_MODEL_PATH)

    server = start_sglang_server(
        model_path=_MODEL_PATH,
        enable_deterministic_inference=True,
        extra_args=[
            "--attention-backend",
            "fa3",
            "--reasoning-parser",
            "qwen3",
            "--tool-call-parser",
            "qwen25",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.5",
        ],
    )
    try:
        yield server
    finally:
        server.stop()


def test_qwen3_8b_h200_fa3_agentic_v2_same_prompt_retries_match_v1_training_payload_bitwise(sglang_server):
    # [FUNCTION DOC] The test. Runs the agentic retry trajectories against the shared backend once
    # with version 1 and once with version 2, each with freshly built input samples and the model
    # path as checkpoint. Asserts both result lists have 16 entries; for each pair asserts the first
    # sample's index and rollout id equal the position in both, then calls the shared parity
    # assertion helper to compare the trajectories (bitwise training payload parity).
    v1_runs = run_agentic_retry_trajectories(
        backend_url=sglang_server.base_url,
        hf_checkpoint=_MODEL_PATH,
        version=V1,
        input_samples=_build_input_samples(),
    )
    v2_runs = run_agentic_retry_trajectories(
        backend_url=sglang_server.base_url,
        hf_checkpoint=_MODEL_PATH,
        version=V2,
        input_samples=_build_input_samples(),
    )

    assert len(v1_runs) == len(v2_runs) == _BATCH_SIZE
    for index, (v1, v2) in enumerate(zip(v1_runs, v2_runs, strict=True)):
        assert v1.samples[0].index == v2.samples[0].index == index
        assert v1.samples[0].rollout_id == v2.samples[0].rollout_id == index
        assert_agentic_retry_trajectory_parity(v1, v2)


def _build_input_samples() -> list[Sample]:
    # [FUNCTION DOC] HELPER: Builds 16 samples whose index and rollout id equal their position,
    # whose prompt is the shared initial agent message list, reward 0.25, and metadata marking the
    # source as parity. Returns the list.
    return [
        Sample(
            index=index,
            rollout_id=index,
            prompt=build_initial_messages(),
            reward=0.25,
            metadata={"source": "parity"},
        )
        for index in range(_BATCH_SIZE)
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
