# [MODULE DOC]
# Tests (helm rendering of the run chart, needs helm) that the chart grants platform accounts
# only to releases containing a platform client: a worker that reads pods gets the read account
# (get, list, watch on pods), a worker that also deletes pods gets the read-delete account (adds
# delete on pods and job creation), a reader does not get the read-delete role, plain workers get
# no service accounts, roles or bindings, and the orchestrator release still gets read-delete.
# Module data: sample worker entries for a reader, a deleter and a plain worker.
import json
from typing import Any

from tests.fast.charts.utils import (
    RUN_ORCHESTRATOR_NAME,
    RUN_PLATFORM_READ_DELETE_NAME,
    RUN_PLATFORM_READ_NAME,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    pod_spec_of,
    render_run,
    requires_helm,
    with_object_names,
)

_READER = {
    "name": "inference-registration-reporter",
    "command": ["x"],
    "serviceAccountName": RUN_PLATFORM_READ_NAME,
}
_DELETER = {
    "name": "trainer-controller-actor",
    "command": ["x"],
    "serviceAccountName": RUN_PLATFORM_READ_DELETE_NAME,
}
_PLAIN = {"name": "inference-engine", "command": ["x"]}


def _render_without_orchestrator(*workers: dict[str, Any]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Renders the run chart with an empty orchestrator command (so no
    # orchestrator is rendered) and the given static worker entries (given object names); returns
    # the parsed objects.
    return render_run(
        "--set-json",
        "run.orchestrator.command=[]",
        "--set-json",
        f"run.staticWorkers={json.dumps(with_object_names(list(workers)))}",
    )


@requires_helm
class TestTheAccountsAReleaseGrantsItsPlatformClients:
    # [CLASS DOC] Tests of which accounts a release creates for its platform clients.
    def test_grants_the_read_account_to_a_release_whose_client_only_reads_pods(self):
        # [FUNCTION DOC] With a reader worker, the read service account, role binding and role exist;
        # the role has one rule (get, list, watch pods in the core group); and the worker's stateful
        # set runs as the read account.
        """A split run's inference release carries the registration reporter, which lists pods to report them."""
        objects = _render_without_orchestrator(_READER)

        assert named_object(objects, "ServiceAccount", RUN_PLATFORM_READ_NAME)
        assert named_object(objects, "RoleBinding", RUN_PLATFORM_READ_NAME)
        assert named_object(objects, "Role", RUN_PLATFORM_READ_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch"]}
        ]
        assert (
            pod_spec_of(objects, "StatefulSet", f"{RUN_RELEASE_NAME}-miles-run-inference-registration-reporter")[
                "serviceAccountName"
            ]
            == RUN_PLATFORM_READ_NAME
        )

    def test_grants_the_read_delete_account_to_a_release_whose_client_also_deletes_pods(self):
        # [FUNCTION DOC] With a deleter worker, the read-delete account, binding and role exist; the
        # role has two rules (pods with get, list, watch, delete; job creation in the batch group).
        """A split run's trainer release suspends cells by deleting pods, which the read-only account refuses."""
        objects = _render_without_orchestrator(_DELETER)

        assert named_object(objects, "ServiceAccount", RUN_PLATFORM_READ_DELETE_NAME)
        assert named_object(objects, "RoleBinding", RUN_PLATFORM_READ_DELETE_NAME)
        assert named_object(objects, "Role", RUN_PLATFORM_READ_DELETE_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]

    def test_grants_a_reader_none_of_the_rights_the_read_delete_account_carries(self):
        # [FUNCTION DOC] With only a reader worker, no role named as the read-delete role is rendered.
        """A reporter may read pod state without inheriting deletion or job-creation rights."""
        objects = _render_without_orchestrator(_READER)

        assert not any(
            obj["metadata"]["name"] == RUN_PLATFORM_READ_DELETE_NAME for obj in objects_of_kind(objects, "Role")
        )

    def test_grants_nothing_to_a_release_no_platform_client_lives_in(self):
        # [FUNCTION DOC] With only a plain worker, no service accounts, roles or role bindings render.
        """A release of plain engines binds no account, so rights over pods would be handed to nobody."""
        objects = _render_without_orchestrator(_PLAIN)

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert objects_of_kind(objects, "Role") == []
        assert objects_of_kind(objects, "RoleBinding") == []

    def test_still_grants_the_release_that_runs_the_orchestration_script(self):
        # [FUNCTION DOC] A default run renders the read-delete account and role with the two rules
        # above, and the orchestrator stateful set runs as that account.
        """The orchestrator deletes pods to heal cells, and it is the reason this rbac existed at all."""
        objects = render_run()

        assert named_object(objects, "ServiceAccount", RUN_PLATFORM_READ_DELETE_NAME)
        assert named_object(objects, "Role", RUN_PLATFORM_READ_DELETE_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]
        assert (
            pod_spec_of(objects, "StatefulSet", RUN_ORCHESTRATOR_NAME)["serviceAccountName"]
            == RUN_PLATFORM_READ_DELETE_NAME
        )
