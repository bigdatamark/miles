# [MODULE DOC] Stateless helpers shared by command backends: repository root and chart paths,
# Python path assembly, rsync command text, training environment variable assembly and parsing,
# wandb argument text, run id generation, boolean environment parsing, a command-line list
# manipulator, mooncake object store argument and port helpers, a TCP readiness probe, pseudo
# file encoding, model argument parsing, hardware tables and detection, shell and process
# runners, and placeholder substitution for multi-node commands.
from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import platform
import random
import shlex
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX
from miles.utils.object_store_config import (
    MOONCAKE_MASTER_ADDRESS_KEY,
    MOONCAKE_MASTER_PORT,
    compute_mooncake_init_kwargs_vanilla,
)
from miles.utils.workers.argv_utils import parse_declared_args
from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME

if TYPE_CHECKING:
    from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest

logger = logging.getLogger(__name__)

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[4]


def _pythonpath_with_sources(megatron_path: str, *additional_pythonpaths: str | None) -> str:
    # [FUNCTION DOC] HELPER: Builds a path-separator-joined list: the repository root, the Megatron
    # path, then the entries of each additional path string (skipping empty or none) in the given
    # order, then the process's existing Python path. Duplicates are removed keeping the first
    # occurrence.
    entries = [str(repo_base_dir), megatron_path]
    for pythonpath in (*additional_pythonpaths, os.environ.get("PYTHONPATH")):
        if pythonpath:
            entries.extend(pythonpath.split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(entries))


def chart_dir(*, repo_base_dir: str | Path) -> Path:
    # [FUNCTION DOC] Path of the helm chart: charts/<chart name> under the given repository root
    # (keyword only; the parameter shadows the module constant).
    return Path(repo_base_dir) / "charts" / CHART_NAME


def rsync_cmd(path_src: str, path_dst: str) -> str:
    # [FUNCTION DOC] Returns shell text that creates the destination directory (with parents) and
    # then rsyncs the contents of the source directory into it in archive mode with overall
    # progress output. Paths are inserted without quoting.
    return f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}"


def train_env_vars(
    request: ExecuteTrainRequest, backend_env_vars: dict[str, str], *, config: ExecuteTrainConfig
) -> dict[str, str]:
    # [FUNCTION DOC] Builds the environment for a training run, later layers winning: unbuffered
    # Python output; for non-FSDP runs, max device connections 1; the backend's own variables;
    # CUDA core dump settings (enable, progress, generation flags that skip big memory regions, and
    # a dump file path under the config's output directory) when core dumping is requested; the
    # request's extra variables; finally variables parsed from the config's extra-env string.
    return {
        # exported for the submitting client too, but only the runtime env reaches the ray workers
        "PYTHONUNBUFFERED": "1",
        # If setting this in FSDP, the computation communication overlapping may have issues
        **(
            {}
            if request.train_backend_fsdp
            else {
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        ),
        **backend_env_vars,
        **(
            {
                "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
                "CUDA_COREDUMP_FILE": f"{config.output_dir}/cuda_coredump_%h.%p.%t",
            }
            if config.cuda_core_dump
            else {}
        ),
        **request.extra_env_vars,
        **_parse_extra_env_vars(config.extra_env_vars),
    }


def _parse_extra_env_vars(text: str):
    # [FUNCTION DOC] HELPER: Parses an environment variable specification. If the text is valid
    # json, returns the parsed value as is (normally an object). Otherwise splits on single spaces,
    # ignores blank pieces, and builds a mapping from each piece's text before the first equals sign
    # to the text between the first and second equals sign.
    # KNOWN ISSUE: a value containing an equals sign is truncated at it, and a piece without an
    # equals sign raises an index error; the odd conditional "or true" in the comprehension has no
    # effect. Valid json that is not an object (a number, say) is returned unchanged and breaks
    # callers that merge it as a mapping. An empty string is invalid json and gives an empty
    # mapping.
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def get_default_wandb_args(test_file: str, run_name_prefix: str | None = None, run_id: str | None = None):
    # [FUNCTION DOC] Configures the "launcher" logger (side effect). If the WANDB_API_KEY
    # environment variable is unset or empty, logs a skip message and returns an empty string.
    # Otherwise derives a test name from the file's stem (prefixed with the parent directory name
    # and an underscore when the stem is shorter than 6 characters), a run name (the given run id or
    # a generated one, plus an underscore and the GITHUB_COMMIT_NAME variable if set, with the
    # optional prefix and an underscore in front), and returns command-line text enabling wandb with
    # project "miles-<test name>", group set to the run name, the API key in single quotes (so a key
    # containing a quote would break the command; the key also appears in the command line), and
    # random suffix disabled.
    from miles.utils.logging_utils import configure_logger_raw

    configure_logger_raw("launcher")
    if not os.environ.get("WANDB_API_KEY"):
        logger.info("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    test_file = Path(test_file)
    test_name = test_file.stem
    if len(test_name) < 6:
        test_name = f"{test_file.parent.name}_{test_name}"

    wandb_run_name = run_id or create_run_id()
    if (x := os.environ.get("GITHUB_COMMIT_NAME")) is not None:
        wandb_run_name += f"_{x}"
    if (x := run_name_prefix) is not None:
        wandb_run_name = f"{x}_{wandb_run_name}"

    # Use the actual key value from environment to avoid shell expansion issues
    wandb_key = os.environ.get("WANDB_API_KEY")
    return (
        "--use-wandb "
        f"--wandb-project miles-{test_name} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key '{wandb_key}' "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    # [FUNCTION DOC] Returns a run id of the form yymmdd-HHMMSS (current UTC time) followed by a
    # dash and a random three-digit number from 000 to 999. Uses the deprecated naive UTC now
    # call.
    return datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


_warned_bool_env_var_keys = set()

_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY = frozenset({"0", "false", "f", "no", "n", "off"})


# copied from SGLang
def get_bool_env_var(name: str, default: str = "false") -> bool:
    # [FUNCTION DOC] Reads the environment variable (default text "false"), lower-cases it, and
    # returns true for 1, true, t, yes, y, on; false for 0, false, f, no, n, off; any other value
    # logs a warning (once per distinct value) and counts as false.
    # KNOWN ISSUE: the already-warned set stores the value rather than the variable name, so the
    # same bad value on a different variable is not warned about again.
    value = os.getenv(name, default).lower()

    if value not in _TRUTHY and value not in _FALSY:
        if value not in _warned_bool_env_var_keys:
            logger.warning(f"get_bool_env_var({name}) see non-understandable value={value} and treat as false")
        _warned_bool_env_var_keys.add(value)

    return value in _TRUTHY


def get_env_enable_infinite_run():
    # [FUNCTION DOC] Whether the MILES_TEST_ENABLE_INFINITE_RUN environment variable is truthy
    # (default false).
    return get_bool_env_var("MILES_TEST_ENABLE_INFINITE_RUN", "false")


class _ArgvDeclaration(NamedTuple):
    # [CLASS DOC] HELPER: One occurrence of a flag in an argument list: the index of the flag
    # token and its value text.
    index: int
    value: str


class ArgvManipulator:
    # [CLASS DOC] Namespace of static helpers for command-line token lists in which a flag's value
    # is either the next token or attached with an equals sign.
    @staticmethod
    def get(argv: list[str], flag: str) -> list[str]:
        # [FUNCTION DOC] Values of every occurrence of the flag, in order (raises an assertion
        # error if a bare flag is the last token).
        return [declaration.value for declaration in _argv_declarations(argv, flag)]

    @staticmethod
    def get_effective(argv: list[str], flag: str) -> str | None:
        # [FUNCTION DOC] The value of the last occurrence of the flag (the one a parser would
        # keep), or none if absent.
        return values[-1] if (values := ArgvManipulator.get(argv, flag)) else None

    @staticmethod
    def is_defined(argv: list[str], flag: str) -> bool:
        # [FUNCTION DOC] True if any token equals the flag or starts with the flag and an equals
        # sign (does not validate that a value follows).
        return any(token == flag or token.startswith(f"{flag}=") for token in argv)

    @staticmethod
    def set(argv: list[str], flag: str, value: str) -> list[str]:
        # [FUNCTION DOC] Returns a new list in which the effective (last) occurrence of the flag has
        # the given value: replaced in place (next token, or inside the attached form), or, if the
        # flag is absent, appended as two tokens at the end. Earlier duplicate occurrences are left
        # untouched. The input is not modified.
        declarations = _argv_declarations(argv, flag)
        if not declarations:
            return [*argv, flag, value]

        last_index = declarations[-1].index
        rewritten = list(argv)
        if rewritten[last_index] == flag:
            rewritten[last_index + 1] = value
        else:
            rewritten[last_index] = f"{flag}={value}"
        return rewritten


def _argv_declarations(argv: list[str], flag: str) -> list[_ArgvDeclaration]:
    # [FUNCTION DOC] HELPER: Scans the tokens; a token equal to the flag takes the following token
    # as value (assertion error if none follows); a token starting with the flag and an equals sign
    # takes the text after the first equals sign. Returns the occurrences in order.
    declarations: list[_ArgvDeclaration] = []
    for index, token in enumerate(argv):
        if token == flag:
            assert index + 1 < len(argv), f"{flag} is the last argument, so it names no value"
            declarations.append(_ArgvDeclaration(index=index, value=argv[index + 1]))
        elif token.startswith(f"{flag}="):
            declarations.append(_ArgvDeclaration(index=index, value=token.split("=", maxsplit=1)[1]))
    return declarations


MOONCAKE_MASTER_METRICS_PORT = 0
MOONCAKE_MASTER_LOG_PATH = Path("/tmp/mooncake_master.log")


OBJECT_STORE_BACKEND_FLAG = "--object-store-backend"
MOONCAKE_BACKEND_NAME = "mooncake"
MOONCAKE_INIT_KWARGS_FLAG = "--mooncake-store-init-kwargs"


def get_mooncake_master_port(train_argv: list[str]) -> int:
    # [FUNCTION DOC] Port of the mooncake master for a run. Looks up the effective value of the
    # mooncake init kwargs flag; if absent, or its json object lacks the master address key,
    # returns the default master port. Otherwise parses that address with the static worker
    # provider's host-and-port parser (lazy import) and returns its port. Invalid json raises.
    from miles.utils.workers.worker_provider.static import parse_host_and_port

    declared = ArgvManipulator.get_effective(train_argv, MOONCAKE_INIT_KWARGS_FLAG)
    if declared is None:
        return MOONCAKE_MASTER_PORT

    address = json.loads(declared).get(MOONCAKE_MASTER_ADDRESS_KEY)
    if address is None:
        return MOONCAKE_MASTER_PORT

    return parse_host_and_port(address).port


def get_mooncake_object_store_args(master_port: int = MOONCAKE_MASTER_PORT, master_host: str = "127.0.0.1") -> str:
    # [FUNCTION DOC] Returns command-line text selecting the mooncake object store backend and
    # passing init kwargs computed for the given master host (default 127.0.0.1) and port (default
    # the standard master port), serialized to json and shell-quoted; ends with a trailing space.
    init_kwargs = compute_mooncake_init_kwargs_vanilla(host=master_host, master_port=master_port)
    return (
        f"{OBJECT_STORE_BACKEND_FLAG} {MOONCAKE_BACKEND_NAME} "
        f"{MOONCAKE_INIT_KWARGS_FLAG} {shlex.quote(json.dumps(init_kwargs))} "
    )


def _is_tcp_server_ready(host: str, port: int) -> bool:
    # [FUNCTION DOC] HELPER: Tries to open a TCP connection with a one second timeout; true on
    # success (connection closed immediately), false on any operating system error.
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def encode_pseudo_file(text: str) -> str:
    # [FUNCTION DOC] Returns the pseudo-file prefix followed by the base64 of the utf-8 text, so
    # file-typed arguments can carry inline content.
    return PSEUDO_FILE_PREFIX + base64.b64encode(text.encode()).decode()


def compute_model_args_overrides(model_type: str) -> dict[str, object]:
    # [FUNCTION DOC] Loads the model argument string for the model type and parses it with
    # Megatron's argument parser (lazy import), returning a mapping from argument name to the
    # value declared in the string.
    from miles.backends.megatron_utils.megatron_config import get_megatron_arg_parser

    return parse_declared_args(load_model_args(model_type), parser=get_megatron_arg_parser())


NUM_GPUS_OF_HARDWARE = {
    "H100": 8,
    "H200": 8,
    "B200": 8,
    "B300": 8,
    "GB200": 4,
    "GB300": 4,
    "Rubin": 4,
    "MI300X": 8,
    "MI325X": 8,
    "MI350X": 8,
    "MI355X": 8,
}

GENERATION_HARDWARE = {
    "H100": "Hopper",
    "H200": "Hopper",
    "B200": "Blackwell",
    "B300": "Blackwell",
    "GB200": "Blackwell",
    "GB300": "Blackwell",
    "Rubin": "Rubin",
}


def detect_hardware() -> str:
    # [FUNCTION DOC] Imports torch lazily and asserts a CUDA device is visible. On ROCm builds
    # picks the first of MI300X, MI325X, MI350X, MI355X contained in the device name. Otherwise by
    # compute capability: 9.0 gives H200 if device 0 has more than 100 GiB of memory else H100;
    # 10.0 gives GB200 on aarch64 (Grace) else B200; 10.3 gives GB300 on aarch64 else B300; 10.7
    # gives Rubin. Asserts the result is known (message asks for an explicit hardware option).
    # Returns the hardware name.
    """Which NUM_GPUS_OF_HARDWARE entry this node is. Call it where the answer is used: prepare steps run GPU-free."""
    import torch

    assert torch.cuda.is_available(), "no visible GPU to detect the hardware from, pass --hardware explicitly"
    name = torch.cuda.get_device_name()
    if torch.version.hip is not None:
        detected = next((hardware for hardware in ("MI300X", "MI325X", "MI350X", "MI355X") if hardware in name), None)
    else:
        grace = platform.machine() == "aarch64"
        match torch.cuda.get_device_capability():
            case (9, 0):
                detected = "H200" if torch.cuda.get_device_properties(0).total_memory > 100 * 1024**3 else "H100"
            case (10, 0):
                detected = "GB200" if grace else "B200"
            case (10, 3):
                detected = "GB300" if grace else "B300"
            case (10, 7):
                detected = "Rubin"
            case _:
                detected = None
    assert detected is not None, f"cannot tell which hardware {name!r} is, pass --hardware explicitly"
    return detected


_PLACEHOLDERS = ("{{node_rank}}", "{{nnodes}}", "{{master_addr}}", "{{node_ip}}")


def run_shell_command(cmd: str, capture_output: bool = False) -> str | None:
    # [FUNCTION DOC] Logs the command, then runs it with bash -c (no outer shell), failing on a
    # non-zero exit status by raising the called-process error (re-raised after logging captured
    # output when capturing). When capturing, output is text, logged, and its standard output
    # returned; otherwise output goes to the parent's streams and none is returned.
    logger.info(f"EXEC: {cmd}")

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            shell=False,
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            logger.error(f"{e.stdout=} {e.stderr=}")
        raise

    if capture_output:
        logger.info(f"Captured stdout={result.stdout} stderr={result.stderr}")
        return result.stdout
    return None


def run_process(
    argv: list[str], *, capture_output: bool, check: bool, input: str | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] Keyword-only flags for capture and check, optional standard input text and
    # timeout in seconds. Logs the shell-quoted command, runs the argument list directly (no shell)
    # with text mode, and returns the completed process. May raise the called-process error (when
    # check is on) or a timeout error.
    logger.info(f"EXEC: {shlex.join(argv)}")
    return subprocess.run(argv, check=check, capture_output=capture_output, text=True, input=input, timeout=timeout)


def substitute_placeholders(cmd: str, *, node_rank: str, nnodes: str, master_addr: str, node_ip: str) -> str:
    # [FUNCTION DOC] Replaces every occurrence of the four placeholders (node rank, node count,
    # master address, node ip; written in double braces) in the command text with the given
    # strings, in that fixed order, and returns the result. Replacement text is inserted
    # literally (later placeholders inside earlier replacements would also be replaced).
    values = {
        "{{node_rank}}": node_rank,
        "{{nnodes}}": nnodes,
        "{{master_addr}}": master_addr,
        "{{node_ip}}": node_ip,
    }
    for placeholder in _PLACEHOLDERS:
        cmd = cmd.replace(placeholder, values[placeholder])
    return cmd
