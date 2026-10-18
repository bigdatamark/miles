# [MODULE DOC] Tests for the outer serve entrypoint. Part one tests splitting the argv at the first
# separator. Part two runs the entrypoint's main in process with the exec call replaced by a
# recorder, checking that it forwards its own arguments and the environment computed by the specs
# to the inner entrypoint (computed variables override inherited ones, a spec may not overwrite the
# pod identity variable, and the parent interpreter's flags are carried over). Part three spawns a
# real serve subprocess on a free local port and talks to it through an rpc handle: typed calls, argv
# and env visibility, light imports, and restart detection by a stable boot uuid client. Subprocesses
# are always terminated (killed if they do not stop in 10 seconds).
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from tests.fast.utils.workers.import_probe import unexpected_light_entrypoint_imports
from tests.fast.utils.workers.serving.serve_smoke_worker import (
    IMPORTED_MODULES_ENV_VAR,
    POOL_ID,
    RPC_PORT_FLAG,
    SMOKE_EXTRA_ENV_VAR,
    SmokeWorker,
)

from miles.utils.http_utils import find_available_port
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.serving import serve as serve_module
from miles.utils.workers.serving.utils import split_worker_argv
from miles.utils.workers.worker_handle import WorkerUnreachableError

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SPECS_PATH = "tests.fast.utils.workers.serving.serve_smoke_worker.compute_specs"


class TestSplitWorkerArgv:
    # [CLASS DOC] Tests the function splitting an argv into entrypoint arguments and worker arguments.
    def test_splits_on_the_separator(self):
        # [FUNCTION DOC] Arguments before the first double dash and those after are returned as two lists.
        """Everything after -- is worker argv, everything before belongs to the entrypoint."""
        assert split_worker_argv(["--pool-id", "p", "--", "--greeting", "hi"]) == (
            ["--pool-id", "p"],
            ["--greeting", "hi"],
        )

    def test_no_separator_means_no_worker_argv(self):
        # [FUNCTION DOC] Without a separator all arguments are entrypoint arguments and the worker
        # list is empty.
        """Without a separator the whole argv belongs to the entrypoint."""
        assert split_worker_argv(["--pool-id", "p"]) == (["--pool-id", "p"], [])

    def test_later_separators_stay_with_the_worker(self):
        # [FUNCTION DOC] Only the first separator splits; a later one stays inside the worker list.
        """Only the first separator splits, so worker argv may contain its own --."""
        assert split_worker_argv(["--", "a", "--", "b"]) == ([], ["a", "--", "b"])


class TestOuterServeForwarding:
    # [CLASS DOC] Tests in-process runs of the outer main function with the process exec call
    # replaced. The cell index environment variable is set to zero in each test.
    def test_forwards_its_own_argv_and_the_env_the_spec_computed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Runs main with entrypoint args (specs path, pool id) and worker args after a
        # separator. Asserts the recorded exec argv is the current interpreter, the module flag, the
        # inner entrypoint module, the same entrypoint args, the separator and the worker args; and that
        # the recorded env holds the smoke echo variable (comma-joined worker argv) and the pool id.
        """serve exists to put the spec's env on the exec'd image; anything it swallowed would never arrive."""
        captured: dict[str, object] = {}

        def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
            # [FUNCTION DOC] HELPER: Records path, argv and env into the captured mapping; never execs.
            captured.update(path=path, argv=argv, env=env)

        monkeypatch.setattr(serve_module.os, "execve", fake_execve)
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        own_argv = ["--specs", _SPECS_PATH, "--pool-id", POOL_ID]
        worker_argv = [RPC_PORT_FLAG, "9000", "--flag", "value"]
        monkeypatch.setattr(sys, "argv", ["serve.py", *own_argv, "--", *worker_argv])

        serve_module.main()

        assert captured["argv"] == [
            sys.executable,
            "-m",
            "miles.utils.workers.serving.serve_inner",
            *own_argv,
            "--",
            *worker_argv,
        ]
        assert captured["env"]["MILES_SERVE_SMOKE_ENV"] == ",".join(worker_argv)
        assert captured["env"]["MILES_SERVE_SMOKE_POOL_ID"] == POOL_ID

    def test_computed_env_overrides_the_parent_without_dropping_unrelated_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Sets a parent variable that the spec will also compute (value "0" via the
        # extra-env mechanism) plus an unrelated parent variable. Asserts the exec env has the computed
        # value for the first and the inherited value for the second.
        """A computed variable replaces its parent value while unrelated inherited variables survive."""
        captured: dict[str, object] = {}

        def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
            # [FUNCTION DOC] HELPER: Records path, argv and env into the captured mapping; never execs.
            captured.update(path=path, argv=argv, env=env)

        computed_name = "MILES_TEST_COMPUTED_ENV"
        inherited_name = "MILES_TEST_INHERITED_ENV"
        monkeypatch.setattr(serve_module.os, "execve", fake_execve)
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.setenv(SMOKE_EXTRA_ENV_VAR, computed_name)
        monkeypatch.setenv(computed_name, "from-parent")
        monkeypatch.setenv(inherited_name, "kept")
        own_argv = ["--specs", _SPECS_PATH, "--pool-id", POOL_ID]
        monkeypatch.setattr(sys, "argv", ["serve.py", *own_argv, "--", RPC_PORT_FLAG, "9000"])

        serve_module.main()

        env = captured["env"]
        assert isinstance(env, dict)
        assert env[computed_name] == "0"
        assert env[inherited_name] == "kept"

    def test_refuses_a_spec_that_overwrites_the_identity_the_platform_gave_the_pod(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Makes the spec compute the subprocess index variable (the pod identity),
        # with exec replaced by a function that fails. Asserts main raises an assertion error whose
        # message names that variable, before any exec happens.
        """--train-env-vars is unrestricted json, and one of these keys makes every rank claim worker zero."""
        monkeypatch.setattr(serve_module.os, "execve", _refuse_exec)
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.setenv(SMOKE_EXTRA_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR)
        own_argv = ["--specs", _SPECS_PATH, "--pool-id", POOL_ID]
        monkeypatch.setattr(sys, "argv", ["serve.py", *own_argv, "--", RPC_PORT_FLAG, "9000"])

        with pytest.raises(AssertionError, match=SUBPROCESS_INDEX_ENV_VAR):
            serve_module.main()


def _refuse_exec(path: str, argv: list[str], env: dict[str, str]) -> None:
    # [FUNCTION DOC] HELPER: Replacement for the exec call that always raises an assertion error
    # saying a spec overwriting the pod identity must not reach exec.
    raise AssertionError("a spec that overwrites the pod's identity must not reach exec")


class TestOuterServeInterpreterFlags:
    # [CLASS DOC] Tests that interpreter flags of the parent are preserved across the exec.
    def test_the_parent_interpreter_flags_are_carried_into_the_inner_process(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] Fakes the interpreter's original argv to include the optimize flag and an
        # extended option (faulthandler), runs main with exec recorded, and asserts the exec argv puts
        # those flags right after the interpreter path and before the module flag and inner module, then
        # the entrypoint args, separator and worker args; and that the exec path is the interpreter.
        """An -O parent that re-execs a bare interpreter silently switches assertions back on inside the worker."""
        captured: dict[str, object] = {}

        def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
            # [FUNCTION DOC] HELPER: Records path, argv and env into the captured mapping; never execs.
            captured.update(path=path, argv=argv, env=env)

        monkeypatch.setattr(serve_module.os, "execve", fake_execve)
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        own_argv = ["--specs", _SPECS_PATH, "--pool-id", POOL_ID]
        worker_argv = [RPC_PORT_FLAG, "9000"]
        monkeypatch.setattr(
            sys,
            "orig_argv",
            [
                sys.executable,
                "-O",
                "-X",
                "faulthandler",
                "-m",
                "miles.utils.workers.serving.serve",
                *own_argv,
                "--",
                *worker_argv,
            ],
        )
        monkeypatch.setattr(sys, "argv", ["serve.py", *own_argv, "--", *worker_argv])

        serve_module.main()

        assert captured["argv"] == [
            sys.executable,
            "-O",
            "-X",
            "faulthandler",
            "-m",
            "miles.utils.workers.serving.serve_inner",
            *own_argv,
            "--",
            *worker_argv,
        ]
        assert captured["path"] == sys.executable


def _spawn_serve(port: int) -> subprocess.Popen:
    # [FUNCTION DOC] HELPER: Starts the real serve entrypoint as a subprocess with the repo root
    # prepended to the python path, the cell index variable set to zero, the repo root as working
    # directory, and arguments: specs path, pool id, separator, rpc port flag with the given port,
    # and a greeting option "hello". Returns the process handle without waiting.
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env[CELL_INDEX_ENV_VAR] = "0"

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "miles.utils.workers.serving.serve",
            "--specs",
            _SPECS_PATH,
            "--pool-id",
            POOL_ID,
            "--",
            RPC_PORT_FLAG,
            str(port),
            "--greeting",
            "hello",
        ],
        cwd=_REPO_ROOT,
        env=env,
    )


async def _wait_ready_or_die(handle, process: subprocess.Popen, timeout: float = 60.0) -> None:
    # [FUNCTION DOC] HELPER: Repeatedly (until the timeout, default 60 seconds) asserts the subprocess
    # is still running (assertion error with the exit code otherwise) and tries the handle's wait-ready
    # with a 2 second timeout, ignoring unreachable errors. Returns on the first success; raises an
    # assertion error if the deadline passes.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None, f"serve subprocess exited early with code {process.returncode}"
        with contextlib.suppress(WorkerUnreachableError):
            await handle.wait_ready(timeout=2.0)
            return
    raise AssertionError("serve subprocess never became ready")


def _stop(process: subprocess.Popen) -> None:
    # [FUNCTION DOC] HELPER: Terminates the process and waits up to 10 seconds; if it has not exited,
    # kills it and waits up to 10 more seconds (which may raise a timeout error).
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


class TestServeEndToEnd:
    # [CLASS DOC] End-to-end tests with real serve subprocesses on local ports derived from the test
    # process id.
    async def test_serve_call_argv_and_env(self):
        # [FUNCTION DOC] Spawns serve, waits for readiness, and asserts: the add call returns 7; the
        # reported worker argv ends with the greeting option and its value; the echoed environment
        # variable ends with the greeting; and the unexpected-light-imports check on the reported
        # imported modules finds nothing. The process is always stopped.
        """serve.py boots a worker subprocess that answers typed calls with argv and env applied."""
        port = find_available_port(20000 + os.getpid() % 10000)
        process = _spawn_serve(port)

        async with httpx.AsyncClient(trust_env=False) as client:
            handle = RpcWorkerHandle(
                SmokeWorker,
                server_url=f"http://127.0.0.1:{port}",
                http_client=client,
            )
            try:
                await _wait_ready_or_die(handle, process)
                assert await handle.demo_sync(a=3, b=4) == 7
                assert (await handle.report_argv())[-2:] == ["--greeting", "hello"]
                assert (await handle.report_env(name="MILES_SERVE_SMOKE_ENV")).endswith("--greeting,hello")
                reported = await handle.report_env(name=IMPORTED_MODULES_ENV_VAR)
                assert unexpected_light_entrypoint_imports(reported) == []
            finally:
                _stop(process)

    async def test_restart_detected_by_stable_boot_uuid_client(self):
        # [FUNCTION DOC] Starts serve and pins a stable boot uuid handle with one successful call. Stops
        # the process, starts a second one on the same port, waits until a fresh non-stable handle
        # reaches it, then asserts the original handle's next call raises a server-restarted error.
        # Both processes are stopped in the cleanup.
        """A stable-boot-uuid client notices when the serve subprocess is restarted."""
        port = find_available_port(21000 + os.getpid() % 10000)
        first_process = _spawn_serve(port)
        second_process: subprocess.Popen | None = None

        async with httpx.AsyncClient(trust_env=False) as client:
            handle = RpcWorkerHandle(
                SmokeWorker,
                server_url=f"http://127.0.0.1:{port}",
                require_stable_boot_uuid=True,
                http_client=client,
            )
            try:
                await _wait_ready_or_die(handle, first_process)
                assert await handle.demo_sync(a=1, b=1) == 2

                _stop(first_process)
                second_process = _spawn_serve(port)
                fresh_handle = RpcWorkerHandle(
                    SmokeWorker,
                    server_url=f"http://127.0.0.1:{port}",
                    http_client=client,
                )
                await _wait_ready_or_die(fresh_handle, second_process)

                with pytest.raises(ServerRestartedError):
                    await handle.demo_sync(a=1, b=1)
            finally:
                if second_process is not None:
                    _stop(second_process)
                _stop(first_process)
