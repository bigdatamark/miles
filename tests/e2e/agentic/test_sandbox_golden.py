# [MODULE DOC]
# Pytest module with one parametrized case per sandbox provider in the credential table. Each
# case runs the golden-episode smoke script (the task's own reference solution executed through
# the Harbor connector, so no GPU, model or session server is involved) and is skipped unless
# the machine holds that provider's credential, SDK and the harbor package; a running case
# creates a real sandbox and spends quota. Registered as a 600-second CPU test in the stage-b-cpu
# suite (not stage-a, because the GPU stages gate on stage-a and this is network-bound) with the
# agentic label. The smoke script lives at the scripts/sandbox_smoke run file under the
# repository root and is loaded by path.
"""A golden episode on each sandbox provider whose credential is present here.

The golden agent is the task's own reference solution, executed by the
connector, so no GPU, no model and no session server take part: what a pass
proves is exactly the platform round trip -- template/image resolution,
sandbox create, exec, verifier, teardown. Offline tests cannot see that layer,
and ``test_harbor_rollout.py`` covers the layers above it.

This drives ``scripts/sandbox_smoke/run.py`` and adds the CI half: one case per
provider in ``PROVIDER_CREDENTIALS``, each skipped unless this machine has that
provider's credential and SDK. So a provider whose
credential CI does not hold reports as a skip, and starts running the day it
does, with no test change. A case that does run creates a real sandbox and
spends real quota, so pick with ``-k`` when you hold several credentials:

    PYTHONPATH=. python -m pytest tests/e2e/agentic/test_sandbox_golden.py -v
    PYTHONPATH=. python -m pytest tests/e2e/agentic/test_sandbox_golden.py -k modal

Each provider's credential and endpoint variables are documented on
``PROVIDER_CREDENTIALS``; which combinations have been run is the table in
``docs/user-guide/sandbox-providers.md``.
"""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

from tests.ci.ci_register import register_cpu_ci

from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, credential_available

# stage-b-cpu, not stage-a: the GPU stages gate on stage-a, so a network-bound
# case does not belong there. est_time is what a cold provider costs when the
# case does run; a skip is immediate.
register_cpu_ci(est_time=600, suite="stage-b-cpu", labels=["agentic"])

RUN_PY = Path(__file__).resolve().parents[3] / "scripts" / "sandbox_smoke" / "run.py"


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    # [FUNCTION DOC] HELPER: Module-scoped fixture that loads the smoke script as a module named
    # "sandbox_smoke_run" from its file path (needed because the file is named run.py) and
    # executes its top-level code. Returns the module.
    """The golden-smoke script, loaded by path because it is named run.py."""
    spec = importlib.util.spec_from_file_location("sandbox_smoke_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", sorted(PROVIDER_CREDENTIALS))
def test_golden_episode(smoke: ModuleType, backend: str):
    # [FUNCTION DOC] Parametrized over the sorted provider names. Skips if the provider's
    # credential is not available (checking the key-file override from the environment as well as
    # the default path), with a message including the provisioning hint; skips if the provider's
    # SDK or the harbor package cannot be imported; if the provider has a target endpoint
    # setting, prints the provider name, label and the effective value (or its default
    # description) for diagnosis. Then runs the smoke script's main with the harbor connector and
    # this backend and asserts exit status zero. Side effect: creates and tears down a real sandbox.
    spec = PROVIDER_CREDENTIALS[backend]
    # the override too, not just the default path: a machine that keeps its key
    # elsewhere runs rollouts fine and must not report as having no credential
    if not credential_available(spec, arg_path=os.environ.get(spec["file_env_var"], "")):
        pytest.skip(f"no {spec['provider']} credential here; provision it with: {spec['provision_hint']}")
    # lacking the SDK or harbor means this case cannot run, not that it failed
    pytest.importorskip(spec["sdk"], reason=f"{spec['provider']} SDK missing: {spec['sdk_hint']}")
    pytest.importorskip("harbor", reason="harbor missing; install line in scripts/sandbox_smoke/README.md")
    # a wrong endpoint fails as an unexplained AgentError, so name the one in effect
    if spec["target"]:
        var, label, default_desc = spec["target"]
        print(f"{spec['provider']} {label}: {os.environ.get(var) or default_desc}", flush=True)

    # the script prints the reward and exit status it judged, which is the failure detail
    assert smoke.main(["--connector", "harbor", "--backend", backend]) == 0
