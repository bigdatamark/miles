# [MODULE DOC] Provider-independent recipe for per-task Terminal-Bench-2 sandboxes, used by the
# Daytona, E2B and Modal materialization modules. Contains: the env server start command template
# and its builder; reading and flooring of task resource requirements from the task config file;
# ownership labels; a daemon-thread keepalive helper; a run-with-deadline helper; shared
# command-line options for the bake tools; deterministic embedding of the env package source as a
# base64 tarball (size-capped at 45000 bytes); the build layer commands (OS dependencies, uv and a
# managed Python 3.12 venv at /opt/envserver, the embedded env source installed with extra
# packages and mcp pinned below 2, and the task directory downloaded from the pinned-commit GitHub
# tarball with the solution directory excluded); and a server readiness poller. The per-command
# execution timeout is read once at import from TB2_COMMAND_TIMEOUT_S (default 900 s).
"""Provider-agnostic image recipe for per-task Terminal-Bench-2 sandboxes.

TB2 is a per-task-image benchmark: every task pins its official runtime image
in ``task.toml`` (``[environment].docker_image`` — the exact image the TB2
harness itself runs). A sandbox serving this env must therefore be built per
task: **the official task image ⊕ this env's server layer**, one layer, no
DinD. This module owns that recipe as plain shell commands — nothing
provider-specific — so the same layers can back a Dockerfile, a Daytona
declarative build (``tb2_sandbox_daytona``, the sibling module), or another
provider.

  ``server_layer_commands(task_dir)``  shell commands that turn the official
      task image into a combined task+env-server image: a uv-managed Python
      venv at ``/opt/envserver`` running the INSTALLED tbench2_env package
      (source embedded into the build, so the sandbox runs exactly the
      version installed here — no released package needed), plus the task
      directory staged at ``/opt/tb2-tasks/<id>``
      (downloaded from the tasks checkout's pinned-commit GitHub tarball) for
      ``reset(task_id)`` via ``TB2_TASKS_DIR``.
  ``server_cmd()``  starts the env server inside the sandbox. Sets
      ``CAMEL_RUNTIME=true`` so camel's TerminalToolkit runs commands in the
      task image's native environment instead of hijacking PATH with its own
      Python 3.10 ``.initial_env`` venv (real TB2 agents see the image's
      python), and ``TB2_COMMAND_TIMEOUT_S`` for realistic command budgets.

Verifier-asset hygiene (mirrors the official harness's stage-at-verify
model): ``solution/`` is excluded from the staged task directory at build
time (nothing in this env ever reads it — it exists only for oracle runs),
and ``SERVER_CMD`` sets ``TB2_WITHHOLD_TESTS=1`` so the server pulls
``tests/`` into process memory at ``reset()`` and deletes it from disk before
the agent's first action; a pristine copy is staged at ``/tests`` only for
the verify window. Residual risk, shared with the official TB2 harness:
verification necessarily runs inside the same container the agent controlled
(task state — services, git state — is not portable), so a root agent that
tampers with container binaries could still fake a pass.
"""

import base64
import concurrent.futures
import getpass
import gzip
import io
import os
import re
import shlex
import subprocess
import tarfile
import threading
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

# Guard for the ONE payload that must be embedded into the build: the
# tbench2_env package source (embedding the local install is what guarantees
# the sandbox scores with exactly the version validated here).
# The hard ceiling is Daytona's Dockerfile parser: a single line may not
# exceed 65535 bytes ("dockerfile line greater than max allowed size",
# observed on real builds), and base64 inflates by 4/3; 45KB of tar.gz stays
# safely under that. Task directories are never embedded — they download from
# the pinned-SHA GitHub tarball instead (see _task_layer_command).
_MAX_INLINE_TAR_BYTES = 45_000

# What `pip install <dir>` needs from the package checkout; everything else
# (uv.lock, caches) stays out of the image.
_ENV_SRC_ITEMS = (
    "pyproject.toml",
    "README.md",
    "openenv.yaml",
    "__init__.py",
    "client.py",
    "models.py",
    "server",
)

# Per-exec timeout inside the sandbox. Named after the variable the env server
# itself reads (it is tbench2_env's contract, not ours), and owned here because
# this is where the server's command line is built — every backend gets the same
# value without three copies of this getenv.
COMMAND_TIMEOUT_S = int(os.getenv("TB2_COMMAND_TIMEOUT_S", "900"))

# The command to start the env server inside a task sandbox (a provider may
# not run the image CMD — Daytona doesn't). /opt/envserver and /opt/tb2-tasks
# are baked by server_layer_commands.
SERVER_CMD = (
    "TB2_TASKS_DIR=/opt/tb2-tasks "
    "TB2_DEFAULT_TASK_ID={default_task_id} "
    "TB2_COMMAND_TIMEOUT_S={command_timeout_s} "
    "TB2_WITHHOLD_TESTS=1 "
    "CAMEL_RUNTIME=true "
    "MAX_CONCURRENT_ENVS=1 "
    "/opt/envserver/bin/python -m uvicorn tbench2_env.server.app:app "
    "--host 0.0.0.0 --port 8000"
)


def server_cmd(command_timeout_s: int = COMMAND_TIMEOUT_S, default_task_id: str = "") -> str:
    # [FUNCTION DOC] Returns the shell command that starts the env server inside a sandbox, by
    # filling the command template with the per-exec timeout and the default task id (quoted for
    # the shell; an empty id becomes an empty quoted string). The command sets the task dir,
    # default task id, command timeout, test withholding on, camel runtime on, and a maximum of one
    # concurrent env, then runs the server app with the venv python on all interfaces, port 8000.
    # A per-task sandbox stages exactly one task, so make it the default:
    # a reset() with no task_id resolves to the staged task rather than the
    # env's built-in headless-terminal default (which isn't present here).
    return SERVER_CMD.format(
        command_timeout_s=command_timeout_s,
        default_task_id=shlex.quote(default_task_id),
    )


def read_task_config(task_dir: Path) -> dict:
    # [FUNCTION DOC] Parses and returns the task.toml file of the task directory as a nested
    # mapping. Raises a file-not-found error if the file does not exist.
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"{task_dir}: no task.toml")
    return tomllib.loads(toml_path.read_text())


def task_env_resources(task_dir: Path) -> tuple[int, int, int]:
    # [FUNCTION DOC] Reads the environment section of the task config (empty if absent) and
    # returns a triple of CPU count, memory in MB and storage in MB, each converted to an integer
    # (default 1, 2048, 10240) and raised to the floors 1, 2048 and 10240. Raises if the config is
    # missing or a value is not numeric.
    """``(cpus, memory_mb, storage_mb)`` from ``task.toml [environment]``, floored.

    The floors (1 CPU / 2048 MB memory / 10240 MB disk) are recipe policy —
    the env server needs room to run alongside the task — applied once here so
    the providers cannot drift apart on them. Each materialization maps these
    onto its own units and drops what its provider does not take (Modal sizes
    disk itself; E2B sizes everything at template-build time).
    """
    env_cfg = read_task_config(task_dir).get("environment", {})
    return (
        max(1, int(env_cfg.get("cpus", 1))),
        max(2048, int(env_cfg.get("memory_mb", 2048))),
        max(10240, int(env_cfg.get("storage_mb", 10240))),
    )


def sandbox_labels(task_dir: Path) -> dict[str, str]:
    # [FUNCTION DOC] Returns a mapping of ownership labels: "openenv-tbench2-task" = task directory
    # name; "openenv-launcher" = OPENENV_LAUNCHER if set and non-empty, else the local user name
    # (the string "unknown" if it cannot be determined); and, only if OPENENV_RUN_ID is non-empty,
    # "openenv-run-id" = that value. Environment read at call time.
    """Ownership labels/metadata for a per-task sandbox: what it runs, and who
    launched it. Provider-agnostic (Daytona labels, E2B metadata, Modal tags —
    same keys).

    Sandbox APIs record no creator, so in a shared org/deployment these are
    the only attribution. ``openenv-tbench2-task`` keys sweep/cleanup tooling
    to exactly the sandboxes this recipe created (shared orgs run other
    workloads). ``openenv-launcher`` is OPENENV_LAUNCHER when set — do set it
    on shared hosts, where the unix user is a generic account — else the local
    unix user. ``openenv-run-id`` (OPENENV_RUN_ID, optional) additionally
    groups one run's sandboxes for targeted sweeps.
    """
    try:
        user = getpass.getuser()
    except Exception:  # no passwd entry / login env on minimal hosts
        user = "unknown"
    labels = {
        "openenv-tbench2-task": task_dir.name,
        "openenv-launcher": os.environ.get("OPENENV_LAUNCHER") or user,
    }
    run_id = os.environ.get("OPENENV_RUN_ID")
    if run_id:
        labels["openenv-run-id"] = run_id
    return labels


def start_keepalive(beat, thread_name: str, *, interval_s: float, max_consecutive_failures: int) -> None:
    # [FUNCTION DOC] Starts a daemon thread with the given name and returns immediately. The thread
    # loops: sleep for the interval first, then call the beat function; a successful beat resets the
    # failure counter, an exception increments it; the thread ends when the counter reaches the
    # maximum of consecutive failures. It also ends with the process (daemon). No stop handle.
    """Run *beat* every *interval_s* for as long as THIS process lives.

    A daemon thread has exactly the right lifetime for a sandbox keepalive: it
    dies with the process, which is what turns the provider's idle/TTL timer
    into a dead-man's switch for orphans. The thread exits once beats fail
    *max_consecutive_failures* times in a row (the normal case: the episode
    ended and the caller deleted the sandbox).
    """

    def _run() -> None:
        # [FUNCTION DOC] HELPER: Heartbeat thread body; see the enclosing function for the loop.
        failures = 0
        while failures < max_consecutive_failures:
            time.sleep(interval_s)
            try:
                beat()
                failures = 0
            except Exception:
                failures += 1

    threading.Thread(target=_run, name=thread_name, daemon=True).start()


def run_with_deadline(fn, timeout_s: float):
    # [FUNCTION DOC] Runs the zero-argument function on a one-worker thread pool and waits up to
    # the timeout for its result, which is returned; the function's own exception propagates; on
    # timeout the standard timeout error is raised and the pool is shut down without waiting, so
    # the worker keeps running in the background (its result is lost).
    """Run *fn* on a scoped worker thread; raise TimeoutError past *timeout_s*.

    For provider calls that block across many requests with no deadline of
    their own (an E2B template build, a Modal build+create). Deliberately not
    a ``with ThreadPoolExecutor`` block: its ``__exit__`` joins the worker,
    which would wait out the very call the deadline exists to abandon. On
    timeout the provider-side work keeps cooking (and may finish) — reclaiming
    whatever it produces is the caller's provider-specific business — but this
    caller stops holding its locks and semaphore slots.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


def add_task_selection_args(ap) -> None:
    # [FUNCTION DOC] Adds to the given argument parser: a required tasks-dir option and a required
    # mutually exclusive group of a comma-separated tasks option or an all flag. Mutates the parser.
    """The bake CLIs' shared task selection: --tasks-dir + (--tasks | --all)."""
    ap.add_argument("--tasks-dir", required=True, help="local terminal-bench-2 checkout")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--tasks", help="comma-separated task_ids")
    group.add_argument("--all", action="store_true", help="every dir with a task.toml")


def selected_task_ids(args) -> tuple[Path, list[str]]:
    # [FUNCTION DOC] Resolves the tasks dir (home expanded, made absolute) and the task ids:
    # with the all flag, the sorted names of the immediate subdirectories containing a task.toml
    # file; otherwise the comma-separated list trimmed with empties dropped (not checked for
    # existence). Returns the pair of directory and id list.
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    if args.all:
        task_ids = sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").is_file())
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    return tasks_dir, task_ids


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    # [FUNCTION DOC] HELPER: Tar entry filter. Drops (returns none) entries whose base name is a
    # bytecode cache directory or the camel initial-env directory, or ends with a compiled-bytecode
    # or egg-info suffix. Otherwise normalizes the entry for determinism: modification time zero,
    # owner and group ids zero, owner and group names empty; returns the entry.
    name = Path(tarinfo.name).name
    if name in {"__pycache__", ".initial_env"} or name.endswith((".pyc", ".egg-info")):
        return None
    # Zero out metadata the build doesn't need (mtimes, owner): it varies
    # across hosts and checkouts and would break _dir_tar_b64's determinism.
    tarinfo.mtime = 0
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = ""
    return tarinfo


def _dir_tar_b64(paths: list[Path], arcnames: list[str], max_bytes: int) -> str:
    # [FUNCTION DOC] HELPER: Builds a gzip-compressed tar in memory of the given paths stored under
    # the matching archive names (lists must have equal length, else a value error), applying the
    # tar filter and a zero gzip timestamp so the bytes are identical for identical sources. Raises
    # a value error if the compressed size exceeds max_bytes. Returns the base64 text (ASCII).
    # Must be byte-for-byte deterministic for identical source: the b64 is
    # embedded in a build command, so any drift — the gzip header's
    # compression timestamp (mtime=0 suppresses it), per-entry mtimes/owners
    # (_tar_filter zeroes them) — would change the image definition on every
    # call and defeat provider build caches and pre-baked snapshots.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for path, arcname in zip(paths, arcnames, strict=True):
                tar.add(path, arcname=arcname, filter=_tar_filter)
    raw = buf.getvalue()
    if len(raw) > max_bytes:
        raise ValueError(f"embedded tar is {len(raw)} bytes (> {max_bytes}); " "inline embedding not suitable.")
    return base64.b64encode(raw).decode()


def _task_layer_command(task_dir: Path) -> str:
    # [FUNCTION DOC] HELPER: Builds the shell command that stages one task directory. Runs git
    # (subprocess, check on failure) in the parent directory of the task dir to get the current
    # commit hash and the url of the origin remote; parses owner and repository name from a GitHub
    # url (raises a value error if it is not a GitHub remote), forms the archive url of that commit,
    # and returns a command that creates /opt/tb2-tasks, streams the archive through tar,
    # stripping the top-level directory and extracting only this task's directory while excluding
    # its solution subdirectory.
    # KNOWN ISSUE: the repository-name pattern stops at the first dot, so a repository whose name
    # contains a dot would be truncated and give a wrong archive url.
    """Build command that stages the task dir at /opt/tb2-tasks/<id>.

    One uniform path for every task: download the checkout's pinned-commit
    GitHub tarball and extract just this task. Deterministic (the SHA pins
    the content — note: the committed tree, not uncommitted local edits) and
    payload-free, so build commands stay far below provider build-command
    size ceilings (see _MAX_INLINE_TAR_BYTES) regardless of task-dir size.
    Requires the tasks checkout to be a git clone with a GitHub origin.
    """
    repo_root = task_dir.parent
    sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    m = re.search(r"github\.com[:/]+([^/]+)/([^/.]+)", remote)
    if not m:
        raise ValueError(
            f"{task_dir.name}: the tasks checkout's origin ({remote}) is not "
            "a GitHub remote to download the task tarball from."
        )
    owner, repo = m.group(1), m.group(2)
    url = f"https://github.com/{owner}/{repo}/archive/{sha}.tar.gz"
    # solution/ never enters the image: nothing in this env reads it (it
    # exists only for oracle runs), and an agent must not be able to cat the
    # answer out of the staged task directory.
    prefix = f"{repo}-{sha}/{task_dir.name}"
    return (
        f"mkdir -p /opt/tb2-tasks && curl -fsSL {url} | "
        f"tar xz --strip-components=1 -C /opt/tb2-tasks "
        f"--exclude='{prefix}/solution' --exclude='{prefix}/solution/*' '{prefix}'"
    )


def _env_src_dir() -> Path:
    # [FUNCTION DOC] HELPER: Imports the env package on demand and returns the directory holding
    # it; raises a runtime error if that directory has no project file (not an editable/checkout
    # install); import errors propagate if the package is missing.
    """Directory of the installed tbench2_env package source.

    The build embeds the package source into the image (_env_src_tar_b64),
    including pyproject.toml so ``pip install <dir>`` works inside the build.
    That requires tbench2_env to be installed editable from a checkout
    (``pip install -e <OpenEnv>/envs/tbench2_env``), where the package
    directory IS the project directory; a wheel/sdist install ships no
    pyproject.toml, so fail fast here instead of deep inside the image build.
    """
    # In-function, deliberately: only a build that embeds the env source needs
    # the package, and the offline tests import this module without it installed.
    import tbench2_env

    src = Path(tbench2_env.__file__).resolve().parent
    if not (src / "pyproject.toml").is_file():
        raise RuntimeError(
            f"tbench2_env at {src} has no pyproject.toml; the per-task sandbox "
            "build embeds the package source and needs an editable/checkout "
            "install: pip install -e <OpenEnv>/envs/tbench2_env"
        )
    return src


def _env_src_tar_b64() -> str:
    # [FUNCTION DOC] HELPER: Packs the existing items among the fixed list of env package files
    # (project file, readme, yaml descriptor, init, client, models, server directory) from the env
    # source directory under a "tbench2_env_src/" archive prefix, via the deterministic tar helper
    # with the 45000-byte cap, and returns the base64 text. Missing items are skipped silently.
    src_dir = _env_src_dir()
    paths, arcnames = [], []
    for item in _ENV_SRC_ITEMS:
        p = src_dir / item
        if p.exists():
            paths.append(p)
            arcnames.append(f"tbench2_env_src/{item}")
    return _dir_tar_b64(paths, arcnames, _MAX_INLINE_TAR_BYTES)


def resolve_docker_image(task_dir: Path, docker_image: str | None) -> str:
    # [FUNCTION DOC] Returns the explicit image if truthy; otherwise the docker image named in the
    # environment section of the task config. Raises a value error if the config names none (and a
    # file-not-found error if the config is missing).
    if docker_image:
        return docker_image
    docker_image = read_task_config(task_dir).get("environment", {}).get("docker_image")
    if not docker_image:
        raise ValueError(f"{task_dir.name}: task.toml has no [environment].docker_image")
    return docker_image


def server_layer_commands(task_dir: Path) -> list[str]:
    # [FUNCTION DOC] Returns the ordered list of six shell build commands: (1) update apt and
    # install curl, certificates and bash without recommended packages, leaving apt lists in place;
    # (2) install uv into /opt/uv without modifying PATH; (3) create a Python 3.12 venv at
    # /opt/envserver; (4) unpack the embedded env package source (base64 tarball computed now) into
    # /opt/src; (5) install that package plus uvicorn, gradio and mcp below version 2 into the venv;
    # (6) the task staging command. Computing it may run git and read the installed env package,
    # so it can raise the errors of those helpers.
    """The recipe itself: shell commands that turn the OFFICIAL task image
    into a combined task+env-server image. Nothing here is provider-specific —
    the same layers work for docker build, Daytona, Modal, ACA, etc.
    """
    return [
        # Server-layer OS deps. Task images are heterogeneous; assume
        # debian-ish (all 89 current TB2 images are debian/ubuntu based). Do
        # NOT rm /var/lib/apt/lists afterwards: the official task image's apt
        # state is part of the task environment — solutions and agents run
        # bare `apt install` relying on the index the task image baked in.
        # (curl/ca-certificates/bash stay installed: debian-ish images almost
        # always ship them anyway, and the official test.sh apt-installs curl
        # itself at verify time — unlike uv below, no observed task behavior
        # depends on their absence.)
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "--no-install-recommends curl ca-certificates bash",
        # uv + its own managed Python: immune to whatever python (if any)
        # the task base image ships. Installed OUTSIDE PATH (/opt/uv) so the
        # agent's PATH lookup stays faithful to the official task image: no
        # tool the image didn't ship resolves from PATH, and a task image's
        # own uv (e.g. financial-document-processor ships uv 0.8.14 at /bin)
        # is never shadowed. (Filesystem traces under /opt remain visible —
        # a single-container sandbox cannot hide them from a root agent.)
        "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh",
        "/opt/uv/uv venv --python 3.12 /opt/envserver",
        # THIS checkout's tbench2_env source (embedded), not a released
        # package: local fixes (canonical evaluate, TB2_COMMAND_TIMEOUT_S)
        # ship with the image. Deps (openenv, camel-ai, ...) come from PyPI.
        f"mkdir -p /opt/src && echo {_env_src_tar_b64()} | base64 -d | tar xz -C /opt/src",
        # mcp is pinned below 2: camel-ai constrains only mcp>=1.3, mcp 2.0
        # (2026-07-28) dropped the `from mcp.server import FastMCP` re-export
        # its TerminalToolkit imports, and every episode then fails with
        # "camel-ai (TerminalToolkit) is required for TB2".
        "/opt/uv/uv pip install --python /opt/envserver/bin/python /opt/src/tbench2_env_src uvicorn gradio 'mcp<2'",
        # Task directory for reset(task_id) via TB2_TASKS_DIR (pinned-SHA
        # GitHub tarball; see _task_layer_command).
        _task_layer_command(task_dir),
    ]


def wait_server_ready(base_url: str, timeout_s: float = 300.0) -> None:
    # [FUNCTION DOC] Polls the health endpoint under the base url (5 s per request, 2 s pause
    # between attempts) until it returns status 200, then returns. Request errors are remembered
    # and retried. After the timeout (default 300 s) raises a timeout error mentioning the url,
    # the timeout and the last request error. Imports the requests library lazily.
    # In-function, deliberately: requests is not a hard dependency of this
    # module -- the offline tests import it with requests absent -- and only a
    # caller that actually waits on a live server needs it.
    import requests

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(2.0)
    raise TimeoutError(f"env server at {base_url} not ready in {timeout_s}s ({last_err})")
