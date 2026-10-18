# [MODULE DOC] Offline tests of the sandbox smoke harness script (scripts/sandbox_smoke/run.py),
# which is loaded by file path as a module. Checks that every registered connector is callable,
# that command-line axes (connector, backend, agent, task) are passed through to the connector
# runner with the golden default agent and the benchmark preset task, and that a real (non-golden)
# agent requires a model endpoint option. Module constants: the repository root (three levels up
# from this file) and the path of the harness script.
"""Offline checks of the sandbox-smoke harness: the registry's shape and the
axis wiring. The real run needs a sandbox credential and is invoked manually
(see scripts/sandbox_smoke/README.md)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_PY = REPO_ROOT / "scripts" / "sandbox_smoke" / "run.py"


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Builds a module spec from the harness script's
    # file path under the name sandbox_smoke_run, creates the module and executes it (running the
    # script's top-level code, without registering it in the module table); returns the module.
    spec = importlib.util.spec_from_file_location("sandbox_smoke_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_connectors_are_callables(smoke):
    # [FUNCTION DOC] The connector registry contains the harbor connector and every registered
    # runner is callable.
    assert "harbor" in smoke.CONNECTORS
    for name, runner in smoke.CONNECTORS.items():
        assert callable(runner), name


def test_golden_run_wires_all_axes_through(smoke, monkeypatch, tmp_path):
    # [FUNCTION DOC] Replaces the harbor connector with a recording fake, sets the tasks
    # directory environment variable to a temporary path (it overrides the cached clone), and
    # sets the command line to the harbor connector with the daytona backend. Asserts the main
    # function returns zero and the runner saw backend daytona, the preset task fix-git, the golden
    # default agent and the temporary tasks directory. All patches are reverted by monkeypatch.
    """connector/backend/agent/task all reach the runner; the golden default needs no endpoint."""
    seen = {}

    async def fake_runner(tasks_dir, task, *, backend, agent, base_url):
        # [FUNCTION DOC] HELPER: Async fake connector runner; records all its arguments into the
        # shared mapping and returns a result with reward 1.0 and exit status Submitted.
        seen.update(tasks_dir=tasks_dir, task=task, backend=backend, agent=agent, base_url=base_url)
        return {"reward": 1.0, "exit_status": "Submitted"}

    monkeypatch.setitem(smoke.CONNECTORS, "harbor", fake_runner)
    monkeypatch.setenv("TB2_TASKS_DIR", str(tmp_path))  # the env var wins over the cached clone
    monkeypatch.setattr("sys.argv", ["run.py", "--connector", "harbor", "--backend", "daytona"])
    assert smoke.main() == 0
    # the benchmark preset filled in the task; the golden default reaches the connector untranslated
    assert seen["backend"] == "daytona" and seen["task"] == "fix-git" and seen["agent"] == smoke.GOLDEN
    assert seen["tasks_dir"] == tmp_path


def test_a_real_harness_requires_a_model_endpoint(smoke, monkeypatch, capsys):
    # [FUNCTION DOC] Sets the command line to harbor, backend e2b and the mini-swe-agent without a
    # base url. Asserts the main function exits with status code 2 and the captured error output
    # mentions the base-url option.
    monkeypatch.setattr(
        "sys.argv", ["run.py", "--connector", "harbor", "--backend", "e2b", "--agent", "mini-swe-agent"]
    )
    with pytest.raises(SystemExit) as excinfo:
        smoke.main()
    assert excinfo.value.code == 2
    assert "--base-url" in capsys.readouterr().err
