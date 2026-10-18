# [MODULE DOC]
# Tests (helm rendering, needs helm) of the workbench chart's RBAC objects: its own namespaced Role
# and binding with no cluster-scoped objects; the full pinned rule set (least privilege); coverage of
# every object kind a colocated run installs (create) and of every kind a run release owns for the
# uninstaller (delete); the uninstaller account being a fixed-name, delete-and-read-only account;
# name conflicts; precreated accounts when rbac creation is off; superset property relative to roles
# a run creates; no escalate, bind, impersonate or wildcard rights; leader-worker-set rules being
# switchable; and consistent service account naming. Module data: a pool entry per engine and
# trainer, a run argument tuple with auto uninstall enabled, a pairing configuration and a run
# argument tuple that is colocated.
import json
from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    RELEASE_NAME,
    UNINSTALLER_SERVICE_ACCOUNT,
    named_object,
    objects_of_kind,
    pod_spec,
    render,
    render_error,
    render_run,
    requires_helm,
)

_POOL_ENTRIES = {
    name: [{"name": name, "objectName": f"myrun-miles-run-{name}", "replicas": 1, "size": 1, "command": ["python"]}]
    for name in ("engine", "trainer")
}

UNINSTALLABLE_RUN = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(_POOL_ENTRIES['engine'])}",
    "--set-json",
    f"run.trainerEngines={json.dumps(_POOL_ENTRIES['trainer'])}",
    "--set",
    "run.autoUninstall.enabled=true",
)

_PAIRING_CONFIG = {
    "namespace": NAMESPACE,
    "release": "myrun",
    "trainer_pool_id": "trainer",
    "inference_pools": [
        {
            "pool_id": "engine",
            "layout": {
                "num_inference_cells": 1,
                "num_trainer_cells": 1,
                "num_pods_per_inference_cell": 1,
                "num_pods_per_trainer_cell": 1,
                "num_gpus_per_node": 8,
                "num_gpus_per_inference_pod": 8,
                "gpu_offset": 0,
            },
        }
    ],
}

COLOCATED_RUN = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(_POOL_ENTRIES['engine'])}",
    "--set-json",
    f"run.trainerEngines={json.dumps(_POOL_ENTRIES['trainer'])}",
    "--set-json",
    f"run.colocate={json.dumps(_PAIRING_CONFIG)}",
)


def granted_verbs(role: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    # [FUNCTION DOC] HELPER: Flattens a role's rules into a mapping from pair (API group, resource)
    # to the set of verbs. Rules naming several groups or resources contribute to each pair; a pair
    # appearing in several rules keeps only the verbs of the last such rule (later rules replace,
    # not union).
    return {
        (group, resource): set(rule["verbs"])
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
    }


@requires_helm
class TestRbacTemplates:
    # [CLASS DOC] Tests of the rendered RBAC objects.
    def test_the_workbench_gets_its_own_namespaced_role(self):
        # [FUNCTION DOC] The default render has a service account, role and binding named as the
        # release, no cluster roles or bindings; the binding refers to the namespaced role in the
        # RBAC API group and has one service account subject in the test namespace.
        """The account is granted a Role this chart ships, never a cluster-wide role such as the built-in admin."""
        objects = render()
        role = named_object(objects, "Role", RELEASE_NAME)
        binding = named_object(objects, "RoleBinding", RELEASE_NAME)

        assert named_object(objects, "ServiceAccount", RELEASE_NAME)["metadata"]["name"] == RELEASE_NAME
        assert objects_of_kind(objects, "ClusterRole") == []
        assert objects_of_kind(objects, "ClusterRoleBinding") == []
        assert binding["roleRef"] == dict(apiGroup="rbac.authorization.k8s.io", kind="Role", name=RELEASE_NAME)
        assert binding["subjects"] == [dict(kind="ServiceAccount", name=RELEASE_NAME, namespace=NAMESPACE)]
        assert role["metadata"]["name"] == RELEASE_NAME

    def test_the_role_stays_inside_what_installing_miles_run_needs(self):
        # [FUNCTION DOC] The granted verbs per (group, resource) equal exactly: full write verbs on
        # config maps, secrets, service accounts, services, deployments, stateful sets, jobs, roles,
        # role bindings and leader-worker-sets; pods without create; pod exec with create and get;
        # pod logs with get; events and persistent volume claims with get, list and watch. The write
        # set is create, delete, get, list, patch, update, watch.
        """Least privilege is the point of shipping our own Role, so the rule set is pinned in full."""
        write = {"create", "delete", "get", "list", "patch", "update", "watch"}

        assert granted_verbs(named_object(render(), "Role", RELEASE_NAME)) == {
            ("", "configmaps"): write,
            ("", "secrets"): write,
            ("", "serviceaccounts"): write,
            ("", "services"): write,
            ("", "pods"): {"delete", "get", "list", "patch", "update", "watch"},
            ("", "pods/exec"): {"create", "get"},
            ("", "pods/log"): {"get"},
            ("", "events"): {"get", "list", "watch"},
            ("", "persistentvolumeclaims"): {"get", "list", "watch"},
            ("apps", "deployments"): write,
            ("apps", "statefulsets"): write,
            ("batch", "jobs"): write,
            ("rbac.authorization.k8s.io", "roles"): write,
            ("rbac.authorization.k8s.io", "rolebindings"): write,
            ("leaderworkerset.x-k8s.io", "leaderworkersets"): write,
        }

    def test_the_role_covers_every_object_kind_miles_run_installs(self):
        # [FUNCTION DOC] Derives (group, plural resource) pairs from a colocated run render (resource
        # is the lowercased kind plus "s", group from the API version with core mapped to empty) and
        # asserts every pair is granted to the workbench role with the create verb.
        """A kind miles-run renders but the Role omits turns every colocated install into an apiserver rejection."""
        granted = granted_verbs(named_object(render(), "Role", RELEASE_NAME))
        installed = {
            ("" if group in ("", "v1") else group, obj["kind"].lower() + "s")
            for obj in render_run(*COLOCATED_RUN)
            for group in [obj["apiVersion"].rpartition("/")[0]]
        }

        assert installed <= set(granted), sorted(installed - set(granted))
        assert all("create" in granted[key] for key in installed)

    def test_the_uninstaller_account_can_delete_a_run_and_nothing_else(self):
        # [FUNCTION DOC] The uninstaller role's verbs across all rules are exactly get, list and
        # delete, and its resources are exactly the eleven kinds of a run release (config maps,
        # secrets, service accounts, services, pods, deployments, stateful sets, jobs, roles,
        # role bindings, leader-worker-sets).
        """A run's escape job runs as this account, and every verb beyond deletion is one it does not need."""
        role = named_object(render(), "Role", UNINSTALLER_SERVICE_ACCOUNT)
        granted = granted_verbs(role)

        assert {verb for verbs in granted.values() for verb in verbs} == {"get", "list", "delete"}
        assert set(granted) == {
            ("", "configmaps"),
            ("", "secrets"),
            ("", "serviceaccounts"),
            ("", "services"),
            ("", "pods"),
            ("apps", "deployments"),
            ("apps", "statefulsets"),
            ("batch", "jobs"),
            ("rbac.authorization.k8s.io", "roles"),
            ("rbac.authorization.k8s.io", "rolebindings"),
            ("leaderworkerset.x-k8s.io", "leaderworkersets"),
        }

    def test_the_uninstaller_is_one_account_per_namespace_under_a_fixed_name(self):
        # [FUNCTION DOC] With a different object name, the uninstaller service account still has the
        # fixed name in the test namespace, and its binding references the fixed-name role.
        """A run finds it without knowing which workbench release created it, so the name may not be derived."""
        objects = render("--set", "objectName=another-workbench")

        assert (
            named_object(objects, "ServiceAccount", UNINSTALLER_SERVICE_ACCOUNT)["metadata"]["namespace"] == NAMESPACE
        )
        assert named_object(objects, "RoleBinding", UNINSTALLER_SERVICE_ACCOUNT)["roleRef"]["name"] == (
            UNINSTALLER_SERVICE_ACCOUNT
        )

    def test_the_workbench_account_may_not_take_the_uninstaller_name(self):
        # [FUNCTION DOC] Naming the workbench service account like the uninstaller fails rendering
        # with an error mentioning that name.
        """One name cannot be two ServiceAccounts, and the uninstaller has to outlive the release that made it."""
        error = render_error("--set", f"serviceAccount.name={UNINSTALLER_SERVICE_ACCOUNT}")

        assert UNINSTALLER_SERVICE_ACCOUNT in error

    def test_a_precreated_account_may_carry_both_roles_when_the_chart_creates_none(self):
        # [FUNCTION DOC] With rbac creation off and the account name set to the uninstaller name, no
        # service account is rendered and the pod uses that account name.
        """With rbac.create=false the chart creates no account, so one administrator-owned account can serve both."""
        objects = render("--set", "rbac.create=false", "--set", f"serviceAccount.name={UNINSTALLER_SERVICE_ACCOUNT}")

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert pod_spec(objects)["serviceAccountName"] == UNINSTALLER_SERVICE_ACCOUNT

    def test_the_uninstaller_covers_every_kind_a_run_release_owns(self):
        # [FUNCTION DOC] Derives (group, resource) pairs from a run render with auto uninstall on
        # and asserts the uninstaller role grants every pair with the delete verb.
        """helm uninstall stops at the first kind it may not delete, and leaves the release half removed."""
        granted = granted_verbs(named_object(render(), "Role", UNINSTALLER_SERVICE_ACCOUNT))
        installed = {
            ("" if group in ("", "v1") else group, obj["kind"].lower() + "s")
            for obj in render_run(*UNINSTALLABLE_RUN)
            for group in [obj["apiVersion"].rpartition("/")[0]]
        }

        assert installed <= set(granted), sorted(installed - set(granted))
        assert all("delete" in granted[key] for key in installed)

    def test_the_uninstaller_leaderworkerset_rules_follow_the_workbench_ones(self):
        # [FUNCTION DOC] With leader-worker-set rights off, the uninstaller role has no rule in that
        # API group.
        """A cluster without the LWS CRDs cannot grant those rights to either account."""
        role = named_object(render("--set", "rbac.leaderWorkerSets=false"), "Role", UNINSTALLER_SERVICE_ACCOUNT)

        assert "leaderworkerset.x-k8s.io" not in {group for rule in role["rules"] for group in rule["apiGroups"]}

    def test_the_role_is_a_superset_of_the_role_miles_run_asks_it_to_create(self):
        # [FUNCTION DOC] Every role rendered by a colocated run (at least one) has, for each (group,
        # resource), verbs contained in the workbench role's verbs for that pair.
        """Kubernetes refuses a Role or RoleBinding carrying rules its creator does not already hold."""
        granted = granted_verbs(named_object(render(), "Role", RELEASE_NAME))
        created = [granted_verbs(role) for role in objects_of_kind(render_run(*COLOCATED_RUN), "Role")]

        assert created
        for rules in created:
            assert all(verbs <= granted.get(key, set()) for key, verbs in rules.items())

    def test_the_role_can_neither_escalate_nor_reach_cluster_scope(self):
        # [FUNCTION DOC] The role's resources exclude cluster roles and bindings and any wildcard;
        # its verbs exclude escalate, bind, impersonate and wildcard; its API groups exclude wildcard.
        """It may write namespaced RBAC only because it holds those rules; escalate or bind would lift that ceiling."""
        rules = named_object(render(), "Role", RELEASE_NAME)["rules"]
        resources = {resource for rule in rules for resource in rule["resources"]}
        verbs = {verb for rule in rules for verb in rule["verbs"]}

        assert not {"clusterroles", "clusterrolebindings"} & resources
        assert not {"escalate", "bind", "impersonate", "*"} & verbs
        assert "*" not in resources
        assert "*" not in {group for rule in rules for group in rule["apiGroups"]}

    def test_the_leaderworkerset_rules_can_be_turned_off(self):
        # [FUNCTION DOC] With leader-worker-set rights off, the workbench role has no rule in that API
        # group.
        """A cluster without LWS installed cannot grant rights over it, and must still get a workbench."""
        rules = named_object(render("--set", "rbac.leaderWorkerSets=false"), "Role", RELEASE_NAME)["rules"]
        groups = {group for rule in rules for group in rule["apiGroups"]}

        assert "leaderworkerset.x-k8s.io" not in groups

    def test_rbac_create_false_only_references_a_preexisting_account(self):
        # [FUNCTION DOC] With rbac creation off and an account name "preexisting": no service
        # accounts, roles or bindings render, and the pod uses that account name.
        """Strict clusters pre-create the identity; the chart then creates no RBAC object at all."""
        objects = render("--set", "rbac.create=false", "--set", "serviceAccount.name=preexisting")

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert objects_of_kind(objects, "Role") == []
        assert objects_of_kind(objects, "RoleBinding") == []
        assert pod_spec(objects)["serviceAccountName"] == "preexisting"

    def test_an_overridden_service_account_name_is_used_by_every_object(self):
        # [FUNCTION DOC] With a custom account name: the service account exists under it, the
        # binding's first subject uses it, and the pod's service account name is it.
        """A renamed account must stay consistent across the pod and its bindings, or the pod silently loses rights."""
        objects = render("--set", "serviceAccount.name=custom-sa")
        binding = named_object(objects, "RoleBinding", RELEASE_NAME)

        assert named_object(objects, "ServiceAccount", "custom-sa")["metadata"]["name"] == "custom-sa"
        assert binding["subjects"][0]["name"] == "custom-sa"
        assert pod_spec(objects)["serviceAccountName"] == "custom-sa"
