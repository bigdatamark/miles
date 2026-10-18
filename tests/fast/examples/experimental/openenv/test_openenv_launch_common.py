# [MODULE DOC]
# Offline tests of the openenv launcher helper module (imported by bare name): its table of
# provider credential specs must cover exactly the backends the sandbox common module registers,
# and each spec's launcher argument attribute must be declared on the launch args protocol.
"""Offline tests for what the launcher itself owns of the credential wiring.

The contract (path-not-value key supply, address forwarding, SDK preflight)
lives in miles.rollout.agentic.credentials and is tested at
tests/fast/rollout/agentic/test_credentials.py. What stays here is the
launcher's own obligation: every backend the example registers must be wired
to a complete credential spec it can act on.
"""

import openenv_launch_common as launch
import openenv_sandbox_common as common
import pytest


def test_every_backend_has_a_credential_spec():
    # [FUNCTION DOC] Asserts the key set of the launcher's credential table equals the key set of the sandbox common module's agent module table.
    """A backend registered without credential wiring would fail at launch with
    a KeyError instead of telling the operator what to provision."""
    assert set(launch.PROVIDER_CREDENTIALS) == set(common.AGENT_MODULES)


@pytest.mark.parametrize("backend", sorted(launch.PROVIDER_CREDENTIALS))
def test_every_spec_names_a_launcher_arg(backend):
    # [FUNCTION DOC] Parametrized over sorted backend names; asserts the spec's argument attribute
    # name is among the annotated names of the launch args protocol class.
    """The arg the launcher reads must be declared on the shared config Protocol,
    else a launcher can never override the key-file path."""
    assert launch.PROVIDER_CREDENTIALS[backend]["arg_attr"] in launch.LaunchArgs.__annotations__, backend
