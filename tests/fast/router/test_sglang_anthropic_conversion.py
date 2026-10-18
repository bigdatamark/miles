# [MODULE DOC] CPU-tier (stage-a, estimated 60 seconds) contract test that runs SGLang's own
# Anthropic-entrypoint unit tests (utils and serving) in a subprocess from the SGLang checkout,
# so that the conversion behavior this repo depends on stays pinned. Requires the environment
# variable naming the SGLang source root; the checkout is taken to be its parent directory.
import os
import subprocess
import sys
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


_SGLANG_TESTS = (
    Path("test/registered/unit/entrypoints/anthropic/test_utils.py"),
    Path("test/registered/unit/entrypoints/anthropic/test_serving.py"),
)


def test_sglang_anthropic_conversion_contract():
    # [FUNCTION DOC] Reads the SGLang source root from the environment (raises a key error if
    # unset), takes its parent as the repo root, and asserts both SGLang test files exist there
    # (assertion error listing missing paths otherwise). Then runs pytest in a subprocess with the
    # repo root as working directory, bytecode writing disabled, cache provider off and quiet
    # mode, on those two files; the subprocess check raises a called-process error on failure.
    sglang_source_root = Path(os.environ["SGLANG_SOURCE_ROOT"])
    sglang_repo_root = sglang_source_root.parent
    missing_tests = [str(path) for path in _SGLANG_TESTS if not (sglang_repo_root / path).is_file()]
    assert not missing_tests, f"Missing SGLang Anthropic tests: {missing_tests}"

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            *(str(path) for path in _SGLANG_TESTS),
        ],
        cwd=sglang_repo_root,
        env=env,
        check=True,
    )
