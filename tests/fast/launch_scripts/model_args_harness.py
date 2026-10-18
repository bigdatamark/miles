# [MODULE DOC]
# HELPER: Test harness exposing model argument scripts to launch-script tests. Lists the model types
# available as python scripts in the scripts/models directory of the repo and expands a model type to
# the list of command line tokens its script produces, for comparison with golden files.
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.external_utils.model_args_utils import load_model_args

_MODEL_SCRIPT_DIR = REPO_ROOT / "scripts" / "models"


def iter_model_types() -> list[str]:
    # [FUNCTION DOC] HELPER: Returns, sorted, the file stems of all python files directly inside the
    # model scripts directory (despite the name, a list rather than an iterator).
    return sorted(path.stem for path in _MODEL_SCRIPT_DIR.glob("*.py"))


def expand_model_args(model_type: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Loads the model args text for the model type through the external-utils
    # loader and splits it on whitespace into tokens (no shell quoting rules). Errors from the loader
    # propagate.
    """Only the producer changed here; the golden files still hold what the shell era expanded to."""
    return load_model_args(model_type).split()
