# [MODULE DOC]
# Loads megatron model argument strings from the per-model python scripts under the repository's
# scripts/models directory (one file per model type exposing a function that returns the flags as
# text). Provides loaders (plain, sibling-relative, shell-quoted), the moe layer frequency pattern
# helper that model scripts call, and a tiny command line (print the args of the given model type).
# This module registers itself under the short name model_args_utils so model scripts can import it.
import importlib.util
import shlex
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_SCRIPT_DIR = _REPO_ROOT / "scripts" / "models"


# ==================== loading a model script ====================


def load_model_args(model_type: str, model_script_dir: Path | None = None, **kwargs: object) -> str:
    # [FUNCTION DOC] Returns the model args text for a model type. Locates the script named after
    # the type in the given directory (default: the repository's model scripts directory); asserts
    # it exists. Registers this module under its short name in the module table (if not already
    # present) so the script can import it, imports the script from its path under a sanitized
    # unique module name (dots and dashes replaced by underscores), calls its model args function
    # with the extra keyword arguments, and collapses all whitespace (including newlines) to single
    # spaces. Asserts the result is non-empty. Side effects: executes the script's module code and
    # touches the module table.
    """Collapse scripts/models/<model_type>.py to one line; a newline would truncate the shell's read -ra."""
    path = (model_script_dir or _MODEL_SCRIPT_DIR) / f"{model_type}.py"
    assert path.exists(), f"No model args script at {path}"
    sys.modules.setdefault("model_args_utils", sys.modules[__name__])
    module = _import_module_from_path(path, f"miles_model_args_{path.stem.replace('.', '_').replace('-', '_')}")
    args = " ".join(module.model_args(**kwargs).split())
    assert args, f"{path} declared no model args"
    return args


def load_sibling_model_args(model_script: str, model_type: str, **kwargs: object) -> str:
    # [FUNCTION DOC] Same as the main loader but looks for the model type's script in the directory
    # containing the given script file (resolved to an absolute path), so a variant loads its base
    # model from the same checkout.
    """Load the model a variant is derived from, out of the same checkout as the variant itself."""
    return load_model_args(model_type, model_script_dir=Path(model_script).resolve().parent, **kwargs)


def shell_safe_model_args(model_type: str | None) -> str:
    # [FUNCTION DOC] Returns the model args with each whitespace-separated token shell-quoted and
    # re-joined by spaces (so brackets are not globbed); an empty string when the model type is
    # none. Errors of the loader propagate.
    """For callers splicing the args into a command line, where --moe-layer-freq [1,1,1] is a glob."""
    if model_type is None:
        return ""
    return " ".join(shlex.quote(token) for token in load_model_args(model_type).split())


# ==================== what a model script may call ====================


def moe_layer_freq(*, nlayers: int, first_k_dense_replace: int) -> str:
    # [FUNCTION DOC] Builds the bracketed, comma-separated per-layer pattern: zero for each of the
    # first layers (dense; count capped at the layer count) and one for the remaining layers (MoE).
    """Render megatron's --moe-layer-freq pattern: the first K layers dense, the rest MoE."""
    dense = min(first_k_dense_replace, nlayers)
    return "[" + ",".join(["0"] * dense + ["1"] * (nlayers - dense)) + "]"


# ==================== importing a file by path ====================


def _import_module_from_path(path: Path, module_name: str) -> ModuleType:
    # [FUNCTION DOC] HELPER: Creates a module from the file location, temporarily registers it in
    # the module table under the name while executing it, always removes the registration afterwards
    # (even on failure), and returns the executed module. Asserts the spec and loader could be built.
    """Import a python file that is not reachable as a dotted module path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[module_name]
    return module


# ==================== command line ====================


if __name__ == "__main__":
    (_MODEL_TYPE,) = sys.argv[1:]
    print(load_model_args(_MODEL_TYPE))
