# [MODULE DOC]
# Consistency test over the real repository: for every test file registered both for CUDA and for a
# ROCm nightly stage-c suite, the GPU count in the ROCm suite name must equal the GPU count of the
# file's CUDA stage. Registers itself as a CPU test in suite stage-a-cpu. Defines a pattern that
# extracts the number from a ROCm nightly suite name of the form nightly-stage-c-<N>-gpu-<...>.
"""A file's ROCm nightly suite asks for as many GPUs as its CUDA stage.

The external ROCm nightly runs the ``nightly-stage-c-<N>-gpu-*`` suites, where
``<N>`` is the GPU count the test needs, as in a CUDA stage name. When a file
registered for both moves its CUDA registration to a stage with a different GPU
count, ``<N>`` moves with it in the same commit, so the nightly gives the test
the GPUs its CUDA stage does. A new test needs no ROCm registration, and the
unprefixed ROCm PR stage is out of scope.
"""

import re
from pathlib import Path

from tests.ci.ci_register import HWBackend, collect_tests, discover_ci_files, register_cpu_ci
from tests.ci.hardware import CUDA_STAGES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[3]

_ROCM_NIGHTLY = re.compile(r"nightly-stage-c-(\d+)-gpu-")


def test_rocm_nightly_gpu_count_follows_the_cuda_stage(monkeypatch):
    # [FUNCTION DOC] Changes the working directory to the repository root, discovers all CI test
    # files and collects their registrations. Builds a mapping from file name to the GPU count of
    # its CUDA stage (looked up by suite in the CUDA stage table). Then lists, sorted, every ROCm
    # registration whose suite matches the nightly-stage-c pattern, whose file also has a CUDA
    # registration, and whose number differs from the CUDA stage's GPU count; each entry is
    # (file name, suite, text saying the CUDA stage has N GPUs). Asserts the list is empty, with a
    # message asking to update the number together with the CUDA stage. Files with only a ROCm
    # registration, and ROCm suites that do not match the pattern, are ignored.
    # discover_ci_files() globs repo-relative paths; pin cwd to the checkout.
    monkeypatch.chdir(REPO_ROOT)
    registrations = collect_tests(discover_ci_files())
    cuda_gpus = {r.filename: CUDA_STAGES[r.suite].num_gpus for r in registrations if r.backend is HWBackend.CUDA}
    mismatched = sorted(
        (r.filename, r.suite, f"CUDA stage has {cuda_gpus[r.filename]} GPUs")
        for r in registrations
        if r.backend is HWBackend.ROCM
        and (match := _ROCM_NIGHTLY.match(r.suite))
        and r.filename in cuda_gpus
        and int(match.group(1)) != cuda_gpus[r.filename]
    )
    assert mismatched == [], (
        "nightly-stage-c-<N>-gpu-* must use the GPU count of the file's CUDA stage; "
        f"update <N> together with the CUDA stage: {mismatched}"
    )
