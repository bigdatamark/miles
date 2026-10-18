# [MODULE DOC] HELPER: Test utility that loads the external-rollout end-to-end script (the short
# qwen2.5 0.5B external rollout e2e test file, located relative to the repository root) as a module
# so fast tests can inspect its functions without running it as a test.
import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "tests/e2e/short/test_qwen2.5_0.5B_external_rollout.py"


def load_external_rollout_script() -> ModuleType:
    # [FUNCTION DOC] HELPER: Builds a module spec from the script path under the name
    # external_rollout_e2e, creates the module, executes the script's top-level code in it (side
    # effects of that code run), and returns the module. Not registered in the module table.
    spec = importlib.util.spec_from_file_location("external_rollout_e2e", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
