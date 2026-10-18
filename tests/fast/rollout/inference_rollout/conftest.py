# [MODULE DOC] Shared pytest fixtures for the inference rollout tests: a fully parsed arguments
# object built through the real command-line parser from a fixed fake command line (FSDP backend,
# CI test mode, tiny rollout, a math reward model, the project's router at a local address, a small
# Qwen model checkpoint name and a null prompt data path), so tests get realistic defaults.
from unittest.mock import patch

import pytest

from miles.utils.arguments import parse_args


def _build_mock_args(extra_argv: list[str] | None = None):
    # [FUNCTION DOC] HELPER: Builds the fixed command line described in the module header, appends
    # the optional extra arguments (default none), temporarily replaces the process argument vector
    # with it, and returns the namespace produced by the real argument parser. May raise whatever
    # the parser raises (including a system exit on invalid options). Nothing is read from the
    # prompt data path or the checkpoint here.
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--ci-test",
        "--rollout-batch-size",
        "2",
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "4",
        "--rollout-num-gpus-per-engine",
        "2",
        "--hf-checkpoint",
        "Qwen/Qwen3-0.6B",
        "--prompt-data",
        "/dev/null",
        "--input-key",
        "input",
        "--label-key",
        "label",
        "--rm-type",
        "math",
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        "30000",
    ] + (extra_argv or [])
    with patch("sys.argv", argv):
        return parse_args()


@pytest.fixture
def mock_args():
    # [FUNCTION DOC] HELPER: Fixture returning the parsed default mock arguments (no extra
    # options); a fresh object per test.
    return _build_mock_args()
