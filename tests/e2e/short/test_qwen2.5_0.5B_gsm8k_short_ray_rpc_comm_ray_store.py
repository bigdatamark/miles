# [MODULE DOC] Thin end-to-end test variant: runs the shared Qwen2.5-0.5B GSM8K short test
# (loaded from its sibling file by path) with the remote procedure call worker communication
# backend and the Ray object store (instead of Mooncake). Registered for CUDA 8-GPU H100 (est. 450
# s) and ROCm 8-GPU MI350 nightly (est. 300 s), labels short and rpc-comm. Contains no training
# code of its own. Run as a script: loads the base module and calls its entrypoint with this
# file's path.
import importlib.util
from pathlib import Path
from types import ModuleType

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(
    est_time=450,
    suite="stage-c-8-gpu-h100",
    labels=["short", "rpc-comm"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(est_time=300, suite="nightly-stage-c-8-gpu-mi350", labels=["short", "rpc-comm"])


def _load_base_test() -> ModuleType:
    # [FUNCTION DOC] HELPER: Loads the base short GSM8K test module from the sibling file with the
    # same directory, under a private module name, by building a module from the file location and
    # executing it (without registering it in the module table). Executing it re-runs that file's
    # top-level code, including its CI registrations. Returns the module object. Fails with an
    # attribute error if the file cannot be found or has no loader.
    path = Path(__file__).with_name("test_qwen2.5_0.5B_gsm8k_short.py")
    spec = importlib.util.spec_from_file_location("gsm8k_short_base_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    base = _load_base_test()
    base.entrypoint(
        comm_backend=base.WorkerCommBackend.RPC,
        object_store_backend=base.ObjectStoreBackend.RAY,
        test_file=__file__,
    )
