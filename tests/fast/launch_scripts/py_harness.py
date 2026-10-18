# [MODULE DOC]
# HELPER: Harness for running python launch scripts without a cluster and recording what they would
# execute. It discovers launchers and their public entrypoints, freezes the environment (hardware,
# run id, pid, selected environment variables, filesystem existence checks), records shell commands and
# generated pseudo files, imports a launcher by path, calls an entrypoint, and formats the recording as
# snapshot text. Constants: frozen run id and hardware, a fake gpu count large enough to satisfy any
# wait loop, frozen pid values, a mapping of frozen environment values, and a tuple of environment
# names cleared before each recording.
import ast
import inspect
import os
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from tests.fast.launch_scripts.sh_harness import REPO_ROOT, sanitize
from tests.fast.utils.command_recorder import record_commands

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.model_args_utils import _import_module_from_path

FROZEN_RUN_ID = "260101-000000-000"
FROZEN_HARDWARE = "H200"

_GPU_COUNT_ANY_WAIT_LOOP_ACCEPTS = "1000000"
_FROZEN_PID = 1000
_FROZEN_PPID = 1001

_FROZEN_ENV = {
    "MASTER_ADDR": "127.0.0.1",
    "MILES_SCRIPT_ENABLE_RAY_SUBMIT": "1",
    "PYTHONPATH": "/frozen/pythonpath",
    "WANDB_API_KEY": "frozen-wandb-api-key",
}

CLEARED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "GITHUB_COMMIT_NAME",
    "GLOO_SOCKET_IFNAME",
    "KEEP_MOE_LORA",
    "MILES_SCRIPT_EXTERNAL_RAY",
    "MILES_USE_LEGACY_ROLLOUT_V1",
    "MLP_SOCKET_IFNAME",
    "MLP_WORKER_0_HOST",
    "MODEL_ARGS_FIRST_K_DENSE_REPLACE",
    "MODEL_ARGS_NUM_LAYERS",
    "MODEL_ARGS_ROTARY_BASE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_FILE",
    "NCCL_NVLS_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "NO_PROXY",
    "OPTIMIZER_CPU_OFFLOAD",
    "RAY_ADDRESS",
    "ROTARY_SCALING_FACTOR",
    "SLURM_JOB_NUM_NODES",
)


@dataclass(frozen=True)
class Recording:
    # [CLASS DOC] HELPER: Immutable container (its lists are mutated in place by recorders) of the shell
    # command strings a launcher issued, in order, and the texts of the pseudo files it encoded.
    commands: list[str]
    pseudo_files: list[str]


@dataclass(frozen=True)
class PyLaunchScript:
    # [CLASS DOC] HELPER: A discovered python launcher: its absolute path and the names of its public
    # entrypoint functions.
    path: Path
    entrypoints: tuple[str, ...]

    @property
    def rel(self) -> str:
        # [FUNCTION DOC] HELPER: Property. The path relative to the repo root, with forward slashes.
        return self.path.relative_to(REPO_ROOT).as_posix()


def iter_py_launch_scripts() -> list[PyLaunchScript]:
    # [FUNCTION DOC] HELPER: Returns launcher descriptors, sorted by path, for every file named run
    # underscore anything dot py anywhere under the scripts directory plus the multi-lora tinker
    # serving example. Each carries its public entrypoint names (see the entrypoint name helper). Parses
    # each file; a syntax error propagates.
    paths = sorted(
        [*(REPO_ROOT / "scripts").rglob("run_*.py"), REPO_ROOT / "examples/multi_lora/serve_qwen3_30b_a3b_tinker.py"]
    )
    return [PyLaunchScript(path=path, entrypoints=tuple(_entrypoint_names(path))) for path in paths]


def launcher_hardware_literals() -> dict[str, tuple[str, ...]]:
    # [FUNCTION DOC] HELPER: For each discovered launcher, parses its source and looks for an annotated
    # assignment whose target is named hardware; the annotation's subscript (a literal type) yields the
    # accepted strings, taken from a tuple subscript or a single one. Returns a mapping from the
    # launcher's repo-relative path to the tuple of accepted values; launchers without such a field are
    # absent. If a file has several such assignments the last wins. Assumes the subscript elements are
    # constants.
    """Every value each launcher's `--hardware` accepts, for the launchers that have the flag at all."""
    literals = {}
    for script in iter_py_launch_scripts():
        for node in ast.walk(ast.parse(script.path.read_text())):
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "hardware":
                values = node.annotation.slice
                literals[script.rel] = tuple(
                    e.value for e in (values.elts if isinstance(values, ast.Tuple) else [values])
                )
    return literals


def iter_self_executing_launchers() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns, sorted, the python files under the scripts, examples and tools
    # directories that are not conventional launchers (see the discovery function) but whose text
    # contains the ray job submit phrase (undecodable bytes replaced). Reads every python file under
    # those roots.
    """Launchers that reach the shell themselves rather than through command_utils."""
    roots = [REPO_ROOT / root for root in ("scripts", "examples", "tools")]
    convention = {script.path for script in iter_py_launch_scripts()}
    return sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
        if path not in convention and "ray job submit" in path.read_text(errors="replace")
    )


def install_shell_recorder(monkeypatch, sandbox: Path) -> Recording:
    # [FUNCTION DOC] HELPER: Returns an empty recording and patches (through the monkeypatch) the
    # process-wide environment and stdlib so launchers that call subprocess directly are recorded: sets
    # the log directory variable to the sandbox; replaces the subprocess run function with a fake that
    # appends the command (a string as is, a sequence joined by spaces) to the recording and returns a
    # successful result whose stdout is a huge gpu count; makes sleep a no-op; makes directory creation
    # a no-op; and freezes the pid and parent pid. The fake run's own extra arguments are ignored.
    """A launcher holding its own subprocess handle never touches the recorded command_utils helpers."""
    recording = Recording(commands=[], pseudo_files=[])

    def fake_run(command, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Records the command and returns a completed-process result with
        # return code zero, the huge gpu count as stdout and empty stderr.
        recording.commands.append(command if isinstance(command, str) else " ".join(command))
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=_GPU_COUNT_ANY_WAIT_LOOP_ACCEPTS, stderr=""
        )

    monkeypatch.setenv("MILES_LOG_DIR", str(sandbox))
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(os, "makedirs", lambda path, **kwargs: None)
    monkeypatch.setattr(os, "getpid", lambda: _FROZEN_PID)
    monkeypatch.setattr(os, "getppid", lambda: _FROZEN_PPID)

    return recording


def freeze_environment(monkeypatch, hardware: str = FROZEN_HARDWARE) -> None:
    # [FUNCTION DOC] HELPER: Through the monkeypatch (undone at test end): sets each frozen environment
    # variable, deletes each variable of the cleared list if present, and replaces the backend's
    # hardware detection with a function returning the given hardware name (default H200).
    for key, value in _FROZEN_ENV.items():
        monkeypatch.setenv(key, value)
    for key in CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(base_backend, "detect_hardware", lambda: hardware)


def install_command_recorder(monkeypatch) -> Recording:
    # [FUNCTION DOC] HELPER: Starts the command-utils command recorder (its list becomes the recording's
    # commands) and returns the recording. Also replaces the run id creator with one returning the
    # frozen run id, and the pseudo file encoder with a fake that stores each text and returns a
    # placeholder string numbered from one by the number stored so far.
    recording = Recording(commands=record_commands(monkeypatch), pseudo_files=[])

    def fake_encode_pseudo_file(text: str) -> str:
        # [FUNCTION DOC] HELPER: Appends the text to the recording's pseudo files and returns a
        # base64-prefixed placeholder naming its one-based position.
        recording.pseudo_files.append(text)
        return f"base64:<frozen-pseudo-file-{len(recording.pseudo_files)}>"

    monkeypatch.setattr(command_utils, "create_run_id", lambda: FROZEN_RUN_ID)
    monkeypatch.setattr(command_utils, "encode_pseudo_file", fake_encode_pseudo_file)

    return recording


def import_launch_script(path: Path) -> ModuleType:
    # [FUNCTION DOC] HELPER: Imports the launcher file by path under a unique module name made of a
    # fixed prefix plus the repo-relative path without suffix with slashes turned into underscores.
    # Executes the file's top level code (side effect) and returns the module.
    name = "miles_launch_script_" + path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", "_")
    return _import_module_from_path(path, name)


@contextmanager
def host_filesystem_frozen(sandbox: Path) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager. Globally replaces the path existence check so that paths
    # inside the sandbox or the repo checkout answer truthfully (extra keyword arguments forwarded),
    # and every other path reports not existing. Restores the original on exit even after errors. Only
    # the exists method is patched, not other filesystem probes. Not thread safe.
    """Launchers skip work whose artifact already exists, so only the checkout and the sandbox may be visible.

    Without this the recording depends on which checkpoints the machine happens to carry, and on
    python 3.11 a `/root` path the user cannot stat raises PermissionError instead of reporting
    absence. The checkout stays visible because a launcher legitimately resolves its own model args
    script out of it.
    """
    visible_roots = (sandbox, REPO_ROOT)
    real_exists = Path.exists

    def exists(self: Path, **kwargs: object) -> bool:
        # [FUNCTION DOC] HELPER: Replacement existence check described by the enclosing context manager.
        if any(self == root or self.is_relative_to(root) for root in visible_roots):
            return real_exists(self, **kwargs)
        return False

    Path.exists = exists
    try:
        yield
    finally:
        Path.exists = real_exists


def call_entrypoint(module: ModuleType, name: str, overrides: dict[str, object], sandbox: Path) -> None:
    # [FUNCTION DOC] HELPER: Looks up the named function on the module and inspects its first
    # parameter. Inside the frozen-filesystem context, if that parameter is named args, builds the
    # module's script-args class from the overrides and passes it as the only argument; otherwise passes
    # the overrides as keyword arguments. A missing name raises an attribute error. Snapshots the process
    # environment first and, in a finally step, restores it exactly so launcher changes do not leak.
    entrypoint = getattr(module, name)
    first = next(iter(inspect.signature(entrypoint).parameters.values()), None)
    saved_env = dict(os.environ)
    try:
        with host_filesystem_frozen(sandbox):
            if first is not None and first.name == "args":
                entrypoint(module.ScriptArgs(**overrides))
            else:
                entrypoint(**overrides)
    finally:
        # a leaked knob would make later recordings depend on which launcher ran first
        os.environ.clear()
        os.environ.update(saved_env)


def format_recording(recording: Recording, sandbox: Path) -> str:
    # [FUNCTION DOC] HELPER: Renders the recording as snapshot text. Each command becomes a section
    # headed by a three-hash marker and its index, with the sandbox and repo paths replaced by
    # placeholders and each space that precedes a double-dash flag turned into a newline plus two
    # spaces; then each pseudo file becomes a section headed with its one-based index followed by its
    # sanitized text; sections are separated by blank lines. Returns lines joined by newlines (no
    # trailing newline after the join).
    """The generated config files are the training recipe, so a snapshot that omits them proves little."""
    lines = []
    for index, command in enumerate(recording.commands):
        lines.append(f"### {index}")
        lines.append(re.sub(r" (?=--)", "\n  ", sanitize(command, sandbox=sandbox)))
        lines.append("")
    for index, content in enumerate(recording.pseudo_files, start=1):
        lines.append(f"### pseudo file {index}")
        lines.append(sanitize(content, sandbox=sandbox))
        lines.append("")
    return "\n".join(lines)


def _entrypoint_names(path: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Parses the file and returns, in source order, the names of its top-level
    # plain (non-async) functions whose names do not start with an underscore and are not main.
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.name != "main"
    ]
