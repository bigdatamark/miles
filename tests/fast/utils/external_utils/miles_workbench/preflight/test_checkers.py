# [MODULE DOC] Tests for the individual preflight checkers: resource and verb expansion
# (deduplicated, first seen order across rule sets), binary presence on the search path, cluster
# reachability classification, permission checks on subresources via the parent resource and a
# subresource option, role delegation accepting a grant scoped to the exact role name after a
# resource wide denial, namespace listing (cluster-provided objects excluded from the foreign
# list; unserved resource types versus query errors give unknown results), and leader-worker-set
# API and controller query failures (discovery failure unknown, connection failure fail,
# forbidden unknown). The raw kubectl runner is replaced by fakes returning scripted results.
from __future__ import annotations

import subprocess

import pytest

from miles.utils.external_utils.miles_workbench.preflight import checkers as checkers_module
from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    BinaryPresenceChecker,
    ClusterReachableChecker,
    LeaderWorkerSetApiChecker,
    LeaderWorkerSetControllerChecker,
    NamespaceListingChecker,
    ResourceVerb,
    ResourceVerbAvailabilityChecker,
    RoleDelegationChecker,
    Status,
    expand_resource_verbs,
)


class TestExpandResourceVerbs:
    # [CLASS DOC] Test of the expansion of rule mappings into resource and verb pairs.
    def test_overlapping_rule_sets_keep_each_resource_verb_once_in_first_seen_order(self) -> None:
        # [FUNCTION DOC] Two rule sets with a repeated get on pods inside the first, and overlap
        # between the sets, expand to a list without duplicates ordered by first appearance: pods
        # get, pods list, services create (from the first set), then pods watch and services
        # delete (new in the second).
        """Overlapping rule sets keep each resource-verb pair once in first-seen order."""
        first_rules = {
            "pods": ("get", "list", "get"),
            "services": ("create",),
        }
        second_rules = {
            "pods": ("list", "watch"),
            "services": ("create", "delete"),
        }

        resource_verbs = expand_resource_verbs(first_rules, second_rules)

        assert resource_verbs == [
            ResourceVerb(verb="get", resource="pods"),
            ResourceVerb(verb="list", resource="pods"),
            ResourceVerb(verb="create", resource="services"),
            ResourceVerb(verb="watch", resource="pods"),
            ResourceVerb(verb="delete", resource="services"),
        ]


class TestBinaryPresenceChecker:
    # [CLASS DOC] Test of the binary presence checker.
    @pytest.mark.parametrize(("found", "status"), [("/usr/bin/helm", Status.PASS), (None, Status.FAIL)])
    def test_binary_presence_reflects_path_lookup(
        self, monkeypatch: pytest.MonkeyPatch, found: str | None, status: Status
    ) -> None:
        # [FUNCTION DOC] Parametrized over a found helm path (pass) and a missing binary (fail).
        # The search path lookup is faked to return the parameter; the check result equals a
        # result with the expected status and the message saying helm is installed.
        # KNOWN ISSUE: the expected message says helm is installed even for the failing case where
        # the binary is missing.
        """Binary presence passes exactly when the executable is on PATH."""
        monkeypatch.setattr(checkers_module.shutil, "which", lambda binary: found)

        result = BinaryPresenceChecker("helm").check()

        assert result == checkers_module.CheckResult(status=status, message="helm is installed")


class TestClusterReachableChecker:
    # [CLASS DOC] Test of the cluster reachability checker.
    @pytest.mark.parametrize(
        ("returncode", "output", "status"),
        [
            (0, "developer", Status.PASS),
            (1, "authentication failed", Status.FAIL),
        ],
    )
    def test_identity_and_query_outcome_classify_cluster_access(
        self, monkeypatch: pytest.MonkeyPatch, returncode: int, output: str, status: Status
    ) -> None:
        # [FUNCTION DOC] Parametrized over (exit code 0 with an identity output: pass) and (exit
        # code 1 with an authentication failure output: fail). The kubectl result is faked; the
        # checker result has the expected status and a message containing the phrase cluster is
        # reachable in both cases.
        """Reachability distinguishes a successful API response from a failed request."""
        _return_kubectl_result(monkeypatch, returncode=returncode, stdout=output)

        result = ClusterReachableChecker().check()

        assert result.status is status
        assert "cluster is reachable" in result.message


class TestResourceVerbAvailabilityChecker:
    # [CLASS DOC] Test of the permission availability checker.
    def test_subresource_permission_uses_the_parent_resource_and_subresource_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Checking the get verb on the pod log subresource in namespace rl with a
        # faked yes answer passes, and the single kubectl call is the can-i query for get on pods
        # with a subresource option for log and the namespace option.
        """A subresource permission is queried using kubectl's subresource protocol."""
        calls: list[tuple[str, ...]] = []

        def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake raw kubectl runner; records the argument tuple and
            # returns success with output yes.
            calls.append(args)
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="yes", stderr="")

        monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))

        result = ResourceVerbAvailabilityChecker(
            namespace="rl", resource_verb=ResourceVerb(resource="pods/log", verb="get")
        ).check()

        assert result.status is Status.PASS
        assert calls == [("auth", "can-i", "get", "pods", "--subresource=log", "-n", "rl")]


class TestRoleDelegationChecker:
    # [CLASS DOC] Test of the role delegation checker.
    def test_a_named_role_permission_is_accepted_after_the_resource_wide_probe_is_denied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] The fake kubectl answers exit code 1 (denied) to the first query and 0
        # (allowed) to the second, so a bind check for the role named miles passes once the
        # name-scoped query succeeds after the resource-wide one is denied.
        """Role delegation accepts a grant scoped to the exact role name."""
        answers = iter([1, 0])

        def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake raw kubectl runner; returns the next scripted exit code
            # with empty output.
            return subprocess.CompletedProcess(args=list(args), returncode=next(answers), stdout="", stderr="")

        monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))

        result = RoleDelegationChecker(namespace="rl", verb="bind", role="miles").check()

        assert result.status is Status.PASS


def _return_kubectl_result(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int, stdout: str = "", stderr: str = ""
) -> None:
    # [FUNCTION DOC] HELPER: Replaces the checkers module's raw kubectl runner with one that
    # always returns a completed process with the given exit code, standard output and error
    # output (the latter two default to empty).
    def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] HELPER: Fake runner returning the configured fixed result.
        return subprocess.CompletedProcess(args=list(args), returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))


class TestNamespaceListingChecker:
    # [CLASS DOC] Tests of the namespace listing checker, which lists objects not managed by the
    # release and records the foreign ones.
    def test_cluster_provided_namespace_objects_are_not_reported_as_foreign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] The faked listing contains the root certificate config map, the default
        # service account, a default token secret and two user objects. The check passes and the
        # foreign attribute holds only the two user objects (a config map and a secret).
        """Kubernetes-created namespace objects are excluded while foreign objects remain."""
        _return_kubectl_result(
            monkeypatch,
            returncode=0,
            stdout=(
                "configmap/kube-root-ca.crt\n"
                "serviceaccount/default\n"
                "secret/default-token-7qk2m\n"
                "configmap/foreign-config\n"
                "secret/foreign-secret\n"
            ),
        )
        checker = NamespaceListingChecker(
            namespace="rl", kind="all", selector="managed-by!=Helm", release="miles-workbench"
        )

        result = checker.check()

        assert result.status is Status.PASS
        assert checker.foreign == ("configmap/foreign-config", "secret/foreign-secret")

    @pytest.mark.parametrize(
        ("error", "expected_message"),
        [
            ("the server doesn't have a resource type leaderworkerset", "does not serve leaderworkerset"),
            ("connection reset by peer", "listing leaderworkerset failed: connection reset by peer"),
        ],
    )
    def test_namespace_listing_failures_distinguish_unserved_resources_from_query_errors(
        self, monkeypatch: pytest.MonkeyPatch, error: str, expected_message: str
    ) -> None:
        # [FUNCTION DOC] Parametrized over a server message that the resource type does not exist
        # (expected message: does not serve leaderworkerset) and a connection reset (expected
        # message: listing leaderworkerset failed with the error text). With exit code 1 and the
        # error on standard error, the check result has unknown status and the expected message.
        """Unserved resources and ordinary listing failures retain distinct unknown explanations."""
        _return_kubectl_result(monkeypatch, returncode=1, stderr=error)
        checker = NamespaceListingChecker(
            namespace="rl", kind="leaderworkerset", selector="managed-by!=Helm", release="miles-workbench"
        )

        result = checker.check()

        assert result.status is Status.UNKNOWN
        assert expected_message in result.message


class TestLeaderWorkerSetCheckers:
    # [CLASS DOC] Tests of the leader-worker-set API and controller checkers.
    @pytest.mark.parametrize(
        ("checker", "error", "expected_status"),
        [
            (LeaderWorkerSetApiChecker(), "discovery unavailable", Status.UNKNOWN),
            (LeaderWorkerSetControllerChecker(), "connection reset by peer", Status.FAIL),
            (LeaderWorkerSetControllerChecker(), "deployments.apps is forbidden (Forbidden)", Status.UNKNOWN),
        ],
    )
    def test_lws_query_failures_preserve_unknown_and_fail_classification(
        self,
        monkeypatch: pytest.MonkeyPatch,
        checker: LeaderWorkerSetApiChecker | LeaderWorkerSetControllerChecker,
        error: str,
        expected_status: Status,
    ) -> None:
        # [FUNCTION DOC] Parametrized over (API checker, discovery unavailable, unknown),
        # (controller checker, connection reset, fail) and (controller checker, forbidden
        # deployments listing, unknown). With the kubectl result faked as exit code 1 and the error
        # text on standard error, the check result has the expected status and its message
        # contains the error text.
        """LWS discovery and controller query failures preserve verifiability classifications."""
        _return_kubectl_result(monkeypatch, returncode=1, stderr=error)

        result = checker.check()

        assert result.status is expected_status
        assert error in result.message
