# [MODULE DOC]
# Tests for the fault-form factory in the ft fault-injection fault_forms module. A fault form is a
# named way to hurt a cell. The factory builds, per cell type (actor/trainer and rollout), the list
# of forms for a cluster backend: Ray gets only api-server inject-fault kills, while Kubernetes
# adds pod deletion (and an in-pod sigkill exec form for rollout engines). Tests check the names
# and order of forms, the posted request, namespace validation and that pod deletion bypasses the
# api server.
import random
from unittest.mock import MagicMock

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import fault_forms
from tests.fast.e2e.ft.fault_injection.utils import NAMESPACE, RUN_ID, api_server_fault_forms, config_of, typed_cell

from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend, DeployComponent


def test_ray_draws_the_in_process_kills_for_a_trainer_cell() -> None:
    # [FUNCTION DOC] Asserts the Ray actor forms are named, in order, one inject-fault form per
    # known failure mode.
    """Ray owns no pods, so the only fault it can be asked for is a kill inside the worker."""
    forms = api_server_fault_forms()["actor"]

    assert [form.name for form in forms] == [f"inject_fault:{one.value}" for one in fault_forms.FAILURE_MODES]


def test_ray_draws_a_sigkill_only_for_a_rollout_cell() -> None:
    # [FUNCTION DOC] Asserts the Ray rollout forms consist of the single sigkill inject-fault form.
    """The engine is a subprocess, and exit, segfault and deadlock are faults only its own code can commit."""
    forms = api_server_fault_forms()["rollout"]

    assert [form.name for form in forms] == [f"inject_fault:{FailureMode.SIGKILL.value}"]


def test_kubernetes_draws_the_kills_plus_pod_deletion_for_a_trainer_cell() -> None:
    # [FUNCTION DOC] Asserts the Kubernetes actor forms are all failure-mode kill forms followed by
    # the pod deletion form.
    """Trainer workers are served over rpc on k8s, so pod deletion joins the kills instead of replacing them."""
    forms_of = fault_forms.create_cell_fault_forms(
        base_url="http://control", config=config_of(ClusterBackend.KUBERNETES)
    )

    forms = forms_of["actor"]

    assert [form.name for form in forms] == [
        *(f"inject_fault:{one.value}" for one in fault_forms.FAILURE_MODES),
        fault_forms.DELETE_POD_FORM_NAME,
    ]


def test_every_kill_is_its_own_form_so_the_draw_stays_uniform() -> None:
    # [FUNCTION DOC] Asserts the Kubernetes actor form count is the number of failure modes plus
    # one (pod deletion).
    """Folding the kills into one form would make pod deletion half of every trainer injection."""
    forms_of = fault_forms.create_cell_fault_forms(
        base_url="http://control", config=config_of(ClusterBackend.KUBERNETES)
    )

    assert len(forms_of["actor"]) == len(fault_forms.FAILURE_MODES) + 1


def test_a_kubernetes_run_without_a_namespace_fails_before_the_soak_starts() -> None:
    # [FUNCTION DOC] Builds Kubernetes forms with an empty namespace; asserts an assertion error
    # saying the namespace is needed.
    """kubectl would otherwise delete pods in whatever namespace the kubeconfig happens to point at."""
    with pytest.raises(AssertionError, match="needs the namespace"):
        fault_forms.create_cell_fault_forms(
            base_url="http://control", config=config_of(ClusterBackend.KUBERNETES, namespace="")
        )


def test_an_inject_fault_form_posts_the_failure_mode_it_was_built_for(monkeypatch) -> None:
    # [FUNCTION DOC] Patches the module's http client with a mock recording posts, finds the
    # segfault form among the Ray actor forms and injects into a typed actor cell. Asserts exactly
    # one post, to the api server's cell inject-fault route for actor-0, with a body carrying mode
    # segfault and sub index zero.
    """The form's name must describe what it actually does, or a soak log explains nothing."""
    posted: list[tuple[str, dict]] = []
    requests = MagicMock()
    requests.post.side_effect = lambda url, json, timeout: posted.append((url, json)) or MagicMock()
    monkeypatch.setattr(fault_forms, "requests", requests)

    forms = api_server_fault_forms()["actor"]
    form = next(one for one in forms if one.name == f"inject_fault:{FailureMode.SEGFAULT.value}")
    form.inject(typed_cell("actor-0", "actor"), random.Random(0))

    assert posted == [("http://control/api/v1/cells/actor-0/inject-fault", {"mode": "segfault", "sub_index": 0})]


def test_the_delete_pod_form_never_reaches_the_api_server(monkeypatch) -> None:
    # [FUNCTION DOC] Patches the pod-deletion helper to record its keyword arguments and the http
    # client to a mock. Builds Kubernetes forms, picks the pod deletion form from the rollout list,
    # and injects into an actor cell. Asserts the helper got the cell id, a release name for the
    # all component of the test run id, and the test namespace, and the http post was never called.
    # KNOWN ISSUE: the form is looked up in the rollout cell-type list but injected into an actor
    # cell, so the cell type does not match the list it came from (works only because the form
    # does not check the type).
    """Routing it through inject-fault would test the production path, not an outsider."""
    seen: list[dict] = []
    monkeypatch.setattr(fault_forms, "delete_one_pod_of_cell", lambda **kwargs: seen.append(kwargs) or "pod")
    requests = MagicMock()
    monkeypatch.setattr(fault_forms, "requests", requests)

    forms_of = fault_forms.create_cell_fault_forms(
        base_url="http://control", config=config_of(ClusterBackend.KUBERNETES)
    )
    cell = typed_cell("actor-0", "actor")
    next(
        one for one in forms_of[fault_forms.ROLLOUT_CELL_TYPE] if one.name == fault_forms.DELETE_POD_FORM_NAME
    ).inject(cell, random.Random(0))

    assert [one["cell_id"] for one in seen] == ["actor-0"]
    assert [one["release"] for one in seen] == [
        ReleaseName(run_id=RUN_ID, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()
    ]
    assert [one["namespace"] for one in seen] == [NAMESPACE]
    requests.post.assert_not_called()


def test_a_kubernetes_engine_can_be_crashed_in_place_as_well_as_deleted() -> None:
    # [FUNCTION DOC] Asserts the Kubernetes rollout forms are, in order, the in-pod sigkill exec
    # form and the pod deletion form.
    """An engine pod has no rpc server to take a kill, so it is reached with kubectl exec or by deletion."""
    forms = fault_forms.create_cell_fault_forms(base_url="http://control", config=config_of(ClusterBackend.KUBERNETES))

    assert [form.name for form in forms[fault_forms.ROLLOUT_CELL_TYPE]] == [
        fault_forms.EXEC_SIGKILL_FORM_NAME,
        fault_forms.DELETE_POD_FORM_NAME,
    ]


def test_ray_gains_no_exec_form() -> None:
    # [FUNCTION DOC] Asserts the Ray rollout forms do not include the in-pod sigkill exec form.
    """There is no pod to reach into, and its engines already take an in-process kill."""
    forms = fault_forms.create_cell_fault_forms(base_url="http://control", config=config_of(ClusterBackend.RAY))

    assert fault_forms.EXEC_SIGKILL_FORM_NAME not in [form.name for form in forms[fault_forms.ROLLOUT_CELL_TYPE]]
