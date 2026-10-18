# [MODULE DOC]
# End-to-end tests of the serve entrypoint (the module run as a subprocess that parses its
# options, imports the spec table, and replaces itself with the real server process). Checks the
# exec chain (pid kept, worker arguments passed after the first separator, environment computed
# from the spec and inherited from the launcher, no heavy runtime imported before the exec) and
# fast failure at startup for bad, missing or failing spec tables and for port conflicts.
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from tests.fast.utils.workers.e2e.env_var_hooks import ENV_VAR_FN_FAILURE_MESSAGE, IMPORTED_MODULES_ENV_VAR
from tests.fast.utils.workers.e2e.harness import (
    POOL_ID,
    READY_TIMEOUT_SECONDS,
    REPO_ROOT,
    RPC_PORT_FLAG,
    ServerProcess,
    port_is_refused,
    reserve_port,
    wait_until_serving,
)
from tests.fast.utils.workers.import_probe import unexpected_light_entrypoint_imports

from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR

SMOKE_MODULE = "tests.fast.utils.workers.e2e.env_var_hooks"
SMOKE_SPECS_PATH = f"{SMOKE_MODULE}.compute_specs"
SMOKE_RAISING_SPECS_PATH = f"{SMOKE_MODULE}.compute_failing_specs"


@pytest.fixture
def spawn_with_specs(state_dir: Path, tmp_path: Path) -> Iterator[Callable[..., ServerProcess]]:
    # [FUNCTION DOC] HELPER: Fixture providing a function that launches the serve entrypoint with
    # a chosen spec-table path. Unlike the generic spawn fixture it does not wait for readiness
    # and supports no extra environment or worker arguments. All launched servers are stopped and
    # killed at teardown.
    started: list[ServerProcess] = []

    def start(specs_path: str) -> ServerProcess:
        # [FUNCTION DOC] HELPER: Reserves a free port and a numbered log file, builds the child
        # environment (current environment, repository root prepended to the python path,
        # unbuffered output, cell index "0"), runs the serve module with the specs path, the pool
        # id, and after a separator the state directory and the rpc port. The process gets its
        # own session; output goes to the log. Returns the process record without waiting.
        port = reserve_port()
        log_path = tmp_path / f"specs-server-{len(started)}.log"

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        env[CELL_INDEX_ENV_VAR] = "0"

        argv = [sys.executable, "-m", "miles.utils.workers.serving.serve"]
        argv += ["--specs", specs_path, "--pool-id", POOL_ID]
        argv += ["--", "--state-dir", str(state_dir), RPC_PORT_FLAG, str(port)]

        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                argv, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
            )

        server = ServerProcess(port=port, process=process, log_path=log_path)
        started.append(server)
        return server

    yield start

    for server in started:
        server.stop()
        server.kill()


class TestExecChain:
    # [CLASS DOC] What the entrypoint passes on to the served process.
    async def test_the_served_process_is_the_spawned_one(self, handle, server):
        # [FUNCTION DOC] Asserts the process id reported by the worker equals the spawned
        # process id.
        """execve keeps the pid, so terminating the spawned process really stops the server."""
        assert await handle.report_pid() == server.process.pid

    async def test_worker_argv_reaches_the_factory(self, handle):
        # [FUNCTION DOC] Asserts the worker's reported argument list contains the state
        # directory option (given after the separator).
        """Everything after -- is handed to the worker factory."""
        argv = await handle.report_argv()
        assert "--state-dir" in argv

    async def test_worker_argv_keeps_its_own_separator(self, spawn, make_handle):
        # [FUNCTION DOC] Spawns a server with extra worker arguments containing a second
        # separator and waits for readiness; asserts the last three reported arguments are the
        # extra ones in order.
        """Only the first -- splits, so worker argv may contain further separators."""
        server = spawn(worker_argv=["--flag", "--", "--inner"])
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        argv = await handle.report_argv()
        assert argv[-3:] == ["--flag", "--", "--inner"]

    async def test_the_spec_computes_its_env_from_the_worker_argv(self, handle):
        # [FUNCTION DOC] Reads the argument-list environment variable that the default spec
        # exports to the worker; asserts it contains the state directory option.
        """The spec is rebuilt from the run's own argv, not from the entrypoint's."""
        recorded = await handle.report_env(name="MILES_E2E_ARGV")
        assert "--state-dir" in recorded

    async def test_no_heavy_runtime_is_imported_before_the_exec(self, spawn_with_specs, make_handle):
        # [FUNCTION DOC] Launches the entrypoint with the spec table whose environment hook
        # records the imported top-level modules, waits until it serves and is ready, reads that
        # variable from the worker, and asserts the probe finds no unexpected heavy modules.
        """LD_PRELOAD applies to the exec'd image, so a runtime loaded before it would miss the spec's env."""
        server = spawn_with_specs(SMOKE_SPECS_PATH)
        wait_until_serving(server)
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        reported = await handle.report_env(name=IMPORTED_MODULES_ENV_VAR)
        assert unexpected_light_entrypoint_imports(reported) == []

    async def test_parent_environment_is_inherited(self, spawn, make_handle):
        # [FUNCTION DOC] Spawns a server with an extra marker environment variable, waits for
        # readiness, asserts the worker sees the variable with its value.
        """Environment from the launcher reaches the worker."""
        server = spawn(extra_env={"MILES_E2E_MARKER": "inherited"})
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        assert await handle.report_env(name="MILES_E2E_MARKER") == "inherited"


class TestStartupFailures:
    # [CLASS DOC] The entrypoint must exit with an error instead of serving when startup input is
    # bad.
    async def test_unknown_specs_path_fails_fast(self, spawn):
        # [FUNCTION DOC] Spawns with an unimportable spec path without waiting; asserts a
        # nonzero exit within 30 s and that the port stays refused.
        """A spec table that cannot be imported exits instead of serving."""
        server = spawn(specs_path="no.such.module.compute_specs", wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

    async def test_missing_specs_argument_is_a_usage_error(self, spawn):
        # [FUNCTION DOC] Runs the entrypoint directly with only the pool id option (60 s limit,
        # output captured). Asserts exit code 2 and that the error text contains "usage". The
        # spawn fixture is requested but unused.
        """argparse rejects a missing --specs with its usage exit code."""
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, "-m", "miles.utils.workers.serving.serve", "--pool-id", POOL_ID],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            timeout=60,
        )

        assert result.returncode == 2
        assert b"usage" in result.stderr.lower()

    async def test_port_conflict_fails_fast(self, spawn, server):
        # [FUNCTION DOC] Spawns a second server on the shared server's port without waiting;
        # asserts it exits nonzero within 30 s while the shared server keeps running.
        """A second server on a taken port exits without disturbing the first."""
        conflicting = spawn(port=server.port, wait=False)
        assert conflicting.wait(timeout=30.0) not in (None, 0)
        assert server.is_running()

    @pytest.mark.parametrize("bad_path", ["no_colon_module", "miles.utils.workers.serving.serve.no_such_attr"])
    async def test_bad_specs_paths_fail_fast(self, spawn, bad_path):
        # [FUNCTION DOC] Parametrized over a path lacking a module-attribute split and a path to
        # a missing attribute of a real module; asserts a nonzero exit within 30 s.
        """Malformed or missing spec-table paths exit rather than serving a broken worker."""
        server = spawn(specs_path=bad_path, wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)

    async def test_unknown_specs_module_fails_fast(self, spawn_with_specs):
        # [FUNCTION DOC] Launches with an unimportable module; asserts a nonzero exit within
        # 30 s, a refused port, and a module-not-found error in the log.
        """A spec table whose module cannot be imported exits instead of serving."""
        server = spawn_with_specs("no.such.module.compute_specs")
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)
        assert "ModuleNotFoundError" in server.logs()

    async def test_missing_specs_attribute_fails_fast(self, spawn_with_specs):
        # [FUNCTION DOC] Launches with a real module and a missing attribute; asserts a nonzero
        # exit, a refused port, and an attribute error naming the attribute in the log.
        """A spec table naming an attribute the module lacks exits instead of serving."""
        server = spawn_with_specs(f"{SMOKE_MODULE}.no_such_attr")
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "AttributeError" in logs
        assert "no_such_attr" in logs

    async def test_a_spec_whose_env_raises_fails_fast(self, spawn_with_specs):
        # [FUNCTION DOC] Launches with the spec table whose environment hook raises; asserts a
        # nonzero exit, a refused port, and a runtime error carrying the hook's message in the
        # log.
        """A spec that cannot compute its env exits instead of serving a worker without it."""
        server = spawn_with_specs(SMOKE_RAISING_SPECS_PATH)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "RuntimeError" in logs
        assert ENV_VAR_FN_FAILURE_MESSAGE in logs
