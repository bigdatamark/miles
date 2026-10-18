# [MODULE DOC]
# HELPER: Harness that runs shell launch scripts in a sandbox with every external command replaced by
# a recording shim script, then reports the recorded invocations. Pieces: discovery of shell launchers
# that mention ray job submit; a frozen environment and a restricted system path; shim generation (a
# bash script per shimmed command that appends one record, made of the pid and the arguments separated
# by unit-separator characters and terminated by a record-separator character, to a capture file, with
# optional canned stdout and a special python shim body); a runner; snapshot comparison with an update
# switch via an environment variable; and path sanitization. Constants include the repo root, the
# placeholder strings for the repo and sandbox, the two separators, the frozen environment, the
# shimmed command names, a huge fake gpu count, a fake ray dashboard process line, canned stdouts for
# date and ps, and the shim script template.
import difflib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT_PLACEHOLDER = "<REPO_ROOT>"
SANDBOX_PLACEHOLDER = "<SANDBOX>"

_ARG_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = "\x1e"

_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

_FROZEN_ENV = {
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "TERM": "dumb",
    "MASTER_ADDR": "127.0.0.1",
    "NODE_RANK": "0",
    "WANDB_KEY": "frozen-wandb-key",
    "WANDB_API_KEY": "frozen-wandb-api-key",
}

_SHIMMED_COMMANDS = (
    "apt",
    "apt-get",
    "awk",
    "curl",
    "date",
    "docker",
    "git",
    "hf",
    "ip",
    "mkdir",
    "nc",
    "nvidia-smi",
    "pip",
    "pip3",
    "pkill",
    "ps",
    "python",
    "python3",
    "ray",
    "rm",
    "rsync",
    "scp",
    "sleep",
    "ssh",
    "torchrun",
    "wget",
)

_GPU_COUNT_LARGER_THAN_ANY_WAIT_LOOP_EXPECTS = "1000000"

_FROZEN_RAY_DASHBOARD_PROCESS = "root 1 0.0 0.0 ray dashboard --node-ip-address=10.0.0.1 --dashboard-port=8265"

_SHIM_STDOUT = {
    "date": "20260101_000000",
    "ps": _FROZEN_RAY_DASHBOARD_PROCESS,
}

_PYTHON_SHIM_BODY = """case "${1:-}" in
*/model_args_utils.py) "$MILES_SH_HARNESS_REAL_PYTHON" "$@" ;;
-c)
    case "$2" in
    *cluster_resources*) printf '%s\\n' 'REPLACE_GPU_COUNT' ;;
    *import*) ;;
    *) "$MILES_SH_HARNESS_REAL_PYTHON" "$@" ;;
    esac
    ;;
esac
""".replace(
    "REPLACE_GPU_COUNT", _GPU_COUNT_LARGER_THAN_ANY_WAIT_LOOP_EXPECTS
)

_SHIM_BODY = {
    "python": _PYTHON_SHIM_BODY,
    "python3": _PYTHON_SHIM_BODY,
}

_SHIM_TEMPLATE = """#!/bin/bash
record="$$${{MILES_SH_HARNESS_ARG_SEP}}{name}"
for arg in "$@"; do
    record="$record$MILES_SH_HARNESS_ARG_SEP$arg"
done
printf '%s%s' "$record" "$MILES_SH_HARNESS_RECORD_SEP" >>"$MILES_SH_HARNESS_CAPTURE"
{stdout_statement}exit 0
"""


def iter_launch_scripts() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns, sorted, all shell files under the scripts and examples directories
    # whose text (undecodable bytes replaced) contains the ray job submit phrase.
    roots = [REPO_ROOT / "scripts", REPO_ROOT / "examples"]
    return sorted(
        path for root in roots for path in root.rglob("*.sh") if "ray job submit" in path.read_text(errors="replace")
    )


@dataclass(frozen=True)
class LaunchScriptRun:
    # [CLASS DOC] HELPER: Immutable result of one launch script run: the recorded shimmed invocations
    # as argument lists (command name first) ordered by process id, sanitized stdout and stderr, and the
    # shell's return code.
    invocations: list[list[str]]
    stdout: str
    stderr: str
    returncode: int

    def invocations_of(self, command: str) -> list[list[str]]:
        # [FUNCTION DOC] HELPER: Returns the recorded invocations whose first element equals the command
        # name, preserving order.
        return [argv for argv in self.invocations if argv[0] == command]

    def ray_job_submit_argv(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Finds ray invocations whose second and third arguments are job and
        # submit; asserts there is exactly one (else an assertion error reporting the count) and returns
        # its full argument list.
        matches = [argv for argv in self.invocations_of("ray") if argv[1:3] == ["job", "submit"]]
        assert len(matches) == 1, f"expected exactly one `ray job submit`, got {len(matches)}"
        return matches[0]


def run_launch_script(
    script: Path,
    sandbox: Path,
    args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> LaunchScriptRun:
    # [FUNCTION DOC] HELPER: Runs the script under bash inside the sandbox and returns the result. Steps:
    # create the sandbox directory, a shim directory (written by the shim writer), an empty capture file
    # and a working directory; build a fully frozen environment (the frozen values, a path with the shim
    # directory ahead of the system path, the capture path, both separators and the real python
    # interpreter path); reject extra environment entries that collide with any frozen key; start bash on
    # the script with the extra args, in a new session, with only the frozen plus extra environment, text
    # pipes for output; wait up to the timeout for output (on timeout kills the whole process group with
    # a kill signal, reaps, and re-raises the timeout error); then waits until no live non-zombie process
    # remains in the script's group, within the same overall deadline (assertion error otherwise); parses
    # the capture file. Stdout and stderr have sandbox and repo paths replaced by placeholders.
    # KNOWN ISSUE: the deadline is computed before the script is started but the post-exit wait shares it,
    # so a script that uses most of the timeout leaves almost no time for background cleanup.
    sandbox.mkdir(parents=True, exist_ok=True)
    fake_bin = sandbox / "fake_bin"
    capture = sandbox / "capture"
    workdir = sandbox / "workdir"
    _write_shims(fake_bin)
    capture.write_bytes(b"")
    workdir.mkdir(exist_ok=True)

    frozen = {
        **_FROZEN_ENV,
        "PATH": f"{fake_bin}:{_SYSTEM_PATH}",
        "MILES_SH_HARNESS_CAPTURE": str(capture),
        "MILES_SH_HARNESS_ARG_SEP": _ARG_SEPARATOR,
        "MILES_SH_HARNESS_RECORD_SEP": _RECORD_SEPARATOR,
        "MILES_SH_HARNESS_REAL_PYTHON": sys.executable,
    }
    _reject_unfreezing(extra_env or {}, frozen=frozen)

    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        ["bash", str(script), *args],
        cwd=workdir,
        env={**frozen, **(extra_env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    _wait_until_the_script_leaves_nothing_running(process.pid, deadline=deadline)

    invocations = _parse_capture(capture.read_text(), sandbox=sandbox)

    return LaunchScriptRun(
        invocations=invocations,
        stdout=sanitize(stdout, sandbox=sandbox),
        stderr=sanitize(stderr, sandbox=sandbox),
        returncode=process.returncode,
    )


SNAPSHOT_UPDATE_ENV_VAR = "MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS"


def assert_matches_snapshot(snapshot: Path, actual: str, subject: str) -> None:
    # [FUNCTION DOC] HELPER: If the update environment variable is set to a non-empty value, creates the
    # parent directories and overwrites the snapshot file with the actual text, then returns without
    # comparing. Otherwise asserts the snapshot file exists (message tells how to regenerate), reads it,
    # and if it differs raises an assertion error that names the subject and includes a unified line
    # diff from expected to actual.
    if os.environ.get(SNAPSHOT_UPDATE_ENV_VAR):
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(actual)
        return

    assert snapshot.exists(), f"missing snapshot for {subject}; regenerate with {SNAPSHOT_UPDATE_ENV_VAR}=1"
    expected = snapshot.read_text()
    if actual != expected:
        # a bare equality on a file-sized string reports nothing a reader can act on, and the
        # machine that regenerated the snapshot is rarely the one that fails on it
        diff = difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), fromfile=f"{snapshot}", tofile="actual", lineterm=""
        )
        raise AssertionError(f"{subject} does not match its snapshot:\n" + "\n".join(diff))


def format_invocations(invocations: list[list[str]]) -> str:
    # [FUNCTION DOC] HELPER: Renders invocations as snapshot text: for each, a header line of three
    # hashes and the zero-based index, one json-encoded line per argument, and a blank line; returns the
    # lines joined by newlines.
    lines = []
    for index, argv in enumerate(invocations):
        lines.append(f"### {index}")
        lines.extend(json.dumps(arg) for arg in argv)
        lines.append("")
    return "\n".join(lines)


def _reject_unfreezing(extra_env: dict[str, str], frozen: dict[str, str]) -> None:
    # [FUNCTION DOC] HELPER: Asserts that no key appears in both mappings; the failure message lists
    # the sorted colliding keys.
    """Silently shadowing PATH or a frozen value would unshim the run or unfreeze the snapshot."""
    collisions = sorted(set(extra_env) & set(frozen))
    assert not collisions, f"extra_env may not override the harness-controlled {collisions}"


def _wait_until_the_script_leaves_nothing_running(pgid: int, deadline: float) -> None:
    # [FUNCTION DOC] HELPER: Polls every 5 milliseconds until no live process remains in the process
    # group; asserts the monotonic clock has not passed the deadline each time some are alive (the
    # failure lists the sorted live pids). Returns nothing.
    """A script backgrounding a shimmed command outlives bash, and would append after we read."""
    while True:
        alive = _live_pids_of_group(pgid)
        if not alive:
            return
        assert time.monotonic() < deadline, f"process group {pgid} still running after the timeout: {sorted(alive)}"
        time.sleep(0.005)


def _live_pids_of_group(pgid: int) -> set[int]:
    # [FUNCTION DOC] HELPER: Linux only. Scans the numeric entries of the process filesystem, reads each
    # stat file (skipping processes that vanish or cannot be read), takes the fields after the last
    # closing parenthesis of the name (so names containing spaces are safe), and returns the pids whose
    # process group equals the argument and whose state is not zombie.
    """A zombie can no longer append to the capture, but killpg still reports its group as alive.

    A command backgrounded inside a command substitution is orphaned when the substitution's
    subshell exits, and nothing reaps it when PID 1 is not an init that does so.
    """
    live = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
        except OSError:
            continue
        state, process_group = fields[0], int(fields[2])
        if process_group == pgid and state != "Z":
            live.add(int(entry.name))
    return live


def _write_shims(fake_bin: Path) -> None:
    # [FUNCTION DOC] HELPER: Creates the directory if needed and writes one executable (mode 755) file
    # per shimmed command from the shim template: it records the name; commands with canned stdout (date,
    # ps) print that quoted value; python and python3 also get the python body that delegates to the real
    # interpreter for the model args utility script, answers cluster resource queries with the huge gpu
    # count, ignores other import-containing one-liners, and runs any remaining one-liner for real.
    fake_bin.mkdir(exist_ok=True)
    for name in _SHIMMED_COMMANDS:
        stdout = _SHIM_STDOUT.get(name)
        stdout_statement = "" if stdout is None else f"printf '%s\\n' {stdout!a}\n"
        shim = fake_bin / name
        shim.write_text(_SHIM_TEMPLATE.format(name=name, stdout_statement=stdout_statement + _SHIM_BODY.get(name, "")))
        shim.chmod(0o755)


def _parse_capture(raw: str, sandbox: Path) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Splits the raw capture on the record separator (dropping empty pieces),
    # splits each record on the argument separator, sorts records by their first field converted to an
    # integer (the pid, so background children appear in pid order rather than append order), and
    # returns each record without the pid with every argument sanitized. A non-numeric first field raises
    # a value error.
    """Order by pid, not by append order: a `&` child appends whenever it gets scheduled."""
    records = [record.split(_ARG_SEPARATOR) for record in raw.split(_RECORD_SEPARATOR) if record != ""]
    records.sort(key=lambda record: int(record[0]))
    return [[sanitize(arg, sandbox=sandbox) for arg in record[1:]] for record in records]


def sanitize(text: str, sandbox: Path) -> str:
    # [FUNCTION DOC] HELPER: Replaces every occurrence of the sandbox path with the sandbox
    # placeholder, then every occurrence of the repo root with the repo placeholder, and returns the
    # text. Order matters if the sandbox lives inside the repo.
    return text.replace(str(sandbox), SANDBOX_PLACEHOLDER).replace(str(REPO_ROOT), REPO_ROOT_PLACEHOLDER)
