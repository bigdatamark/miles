# [MODULE DOC] Tests of the shared cluster-backend availability helpers used to gate tests and
# e2e runs: the kubernetes probe (names the missing kubectl tool, the namespace variable, the
# caller's own namespace source, relays cluster errors, tells that an admin must install the
# leader-worker-set API, asks the API instead of reading the CRD, requires the pod-delete
# permission, bounds every kubectl call with a timeout, and accepts a fully capable cluster), the
# run-level backend creation gate (an unreachable requested backend is a failure), availability
# of a run (namespace from the run config, ray probed by import), the require-backend skip
# helper, and the both-backends decorator. kubectl is faked through a patched process runner.
import subprocess

import pytest
from tests.fast import cluster_backends

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend


class TestKubernetesAvailability:
    # [CLASS DOC] Tests of the kubernetes availability probe's reason messages and call hygiene.
    def test_names_the_missing_tool_rather_than_just_refusing(self, monkeypatch):
        # [FUNCTION DOC] With the tool lookup finding nothing, the unavailability reason mentions
        # kubectl.
        """A developer reading "unavailable" learns nothing; naming kubectl tells them what to install."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: None)

        assert "kubectl" in cluster_backends.kubernetes_availability().reason

    def test_points_at_the_runbook_when_no_namespace_is_set(self, monkeypatch):
        # [FUNCTION DOC] With tools present but the namespace variable unset, the reason names that
        # variable and mentions the cluster-backend runbook.
        """Running in whatever namespace happened to be current is how someone else's run gets deleted."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv(cluster_backends.NAMESPACE_ENV_VAR, raising=False)

        reason = cluster_backends.kubernetes_availability().reason

        assert cluster_backends.NAMESPACE_ENV_VAR in reason
        assert "cluster-backend" in reason

    def test_names_the_variable_the_caller_reads_rather_than_its_own(self, monkeypatch):
        # [FUNCTION DOC] Probing an empty namespace with a caller-supplied namespace source name
        # produces a reason containing that source name and not the default namespace variable.
        """Two callers feed this from two different variables, so a hardcoded hint sends one of them to the wrong knob."""
        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")

        reason = cluster_backends.kubernetes_availability_of_namespace(
            "", namespace_source="MILES_SCRIPT_NAMESPACE"
        ).reason

        assert "MILES_SCRIPT_NAMESPACE" in reason
        assert cluster_backends.NAMESPACE_ENV_VAR not in reason

    def test_reports_what_the_cluster_said_when_it_refuses(self, monkeypatch):
        # [FUNCTION DOC] When every fake kubectl call fails with an Unauthorized error text, the
        # reason contains that text.
        """An expired token and a missing cluster look identical unless the message is passed through."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(returncode=1, stderr="Unauthorized"))

        assert "Unauthorized" in cluster_backends.kubernetes_availability().reason

    def test_says_an_admin_has_to_install_leaderworkerset(self, monkeypatch):
        # [FUNCTION DOC] When only the call involving the leader-worker-set API path fails, the
        # reason mentions an admin.
        """Nothing a developer can do about it, so the message has to say who can."""
        _fake_kubectl(
            monkeypatch,
            lambda argv: _completed(returncode=1 if cluster_backends.LEADER_WORKER_SET_API_PATH in argv else 0),
        )

        assert "admin" in cluster_backends.kubernetes_availability().reason

    def test_asks_the_api_for_leaderworkerset_instead_of_reading_the_crd(self, monkeypatch):
        # [FUNCTION DOC] With all fake calls succeeding, the recorded kubectl argument lists include
        # one with the leader-worker-set API path and none mentioning a CRD or custom resource
        # definitions.
        """The workbench holds a namespaced Role, so a cluster-scoped read reports "not installed" for a cluster that has it."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        cluster_backends.kubernetes_availability()

        assert any(cluster_backends.LEADER_WORKER_SET_API_PATH in argv for argv in calls)
        assert not any("crd" in argv or "customresourcedefinitions" in argv for argv in calls)

    def test_refuses_when_the_account_may_not_delete_pods(self, monkeypatch):
        # [FUNCTION DOC] When the permission check for the delete verb returns empty output, the
        # reason mentions both delete and pods.
        """Injecting a fault deletes a pod, so discovering the missing verb mid-soak wastes the whole run."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="" if "delete" in argv else "yes"))

        reason = cluster_backends.kubernetes_availability().reason

        assert "delete" in reason and "pods" in reason

    def test_bounds_every_call_so_an_unreachable_cluster_cannot_hang_the_probe(self, monkeypatch):
        # [FUNCTION DOC] Replaces the process runner with a fake that records each call's timeout
        # and returns a successful yes answer, sets the namespace to mine, runs the probe, and
        # asserts at least one call happened and every timeout equals the kubectl timeout constant.
        """kubectl against a black-holed endpoint blocks forever, and the probe runs before anything can time it out."""
        timeouts = []

        def fake_run_process(argv, *, capture_output, check, input=None, timeout=None):
            # [FUNCTION DOC] HELPER: Fake process runner; records the timeout argument and returns
            # a completed process with output yes.
            timeouts.append(timeout)
            return _completed(stdout="yes")

        monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "mine")
        monkeypatch.setattr(cluster_backends, "run_process", fake_run_process)

        cluster_backends.kubernetes_availability()

        assert timeouts and all(t == cluster_backends.KUBECTL_TIMEOUT_SECONDS for t in timeouts)

    def test_accepts_a_cluster_that_serves_the_api_and_grants_every_verb(self, monkeypatch):
        # [FUNCTION DOC] With every fake call answering yes, the probe reports available and the
        # reason mentions the configured namespace (mine).
        """The negative cases prove nothing unless the positive one still passes."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        availability = cluster_backends.kubernetes_availability()

        assert availability.available
        assert "mine" in availability.reason


class TestCreateBackendForRun:
    # [CLASS DOC] Tests of the run-level gate that creates a backend only when it is available.
    def test_a_backend_the_lane_asked_for_and_cannot_reach_is_a_failure(self, monkeypatch):
        # [FUNCTION DOC] With the run availability check patched to report unavailable with a given
        # reason, creating the backend for a kubernetes run config raises an assertion error
        # matching that reason.
        """The ft e2e entries are bare python3 with no pytest to skip into, and exiting 0 would report green."""
        monkeypatch.setattr(
            cluster_backends,
            "availability_of_run",
            lambda *, config: cluster_backends.BackendAvailability(False, "no cluster today"),
        )

        with pytest.raises(AssertionError, match="no cluster today"):
            cluster_backends.create_backend_for_run(_run_config(ClusterBackend.KUBERNETES))

    def test_a_reachable_backend_lets_the_run_proceed(self, monkeypatch):
        # [FUNCTION DOC] With the availability check patched to available, creating the backend for
        # a ray run config completes without error (no assertion on the result).
        """The gate must be invisible on the path the run is meant to take."""
        monkeypatch.setattr(
            cluster_backends,
            "availability_of_run",
            lambda *, config: cluster_backends.BackendAvailability(True, "fine"),
        )

        cluster_backends.create_backend_for_run(_run_config(ClusterBackend.RAY))


class TestAvailabilityOfRun:
    # [CLASS DOC] Tests of the availability check of a run from its config.
    def test_the_namespace_comes_from_the_run_config_not_the_fast_tests_variable(self, monkeypatch):
        # [FUNCTION DOC] With the fast tests' namespace variable removed and all fake calls
        # answering yes, a kubernetes run config with namespace miles-e2e is available, and every
        # recorded permission-check call carries that namespace.
        """Regression: the workbench sets MILES_SCRIPT_NAMESPACE only, so reading the other one skipped everything."""
        monkeypatch.delenv(cluster_backends.NAMESPACE_ENV_VAR, raising=False)
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        availability = cluster_backends.availability_of_run(config=_run_config(ClusterBackend.KUBERNETES))

        assert availability.available, availability.reason
        assert all("miles-e2e" in argv for argv in calls if "can-i" in argv)

    def test_a_run_without_a_namespace_names_the_variable_the_run_reads(self, monkeypatch):
        # [FUNCTION DOC] A kubernetes run config with an empty namespace yields a reason that names
        # the run namespace environment variable constant.
        """kubectl would otherwise act on whatever namespace the kubeconfig happens to point at."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(stdout="yes"))

        reason = cluster_backends.availability_of_run(
            config=_run_config(ClusterBackend.KUBERNETES, namespace="")
        ).reason

        assert cluster_backends.RUN_NAMESPACE_ENV_VAR in reason

    def test_a_ray_run_is_probed_by_importing_ray(self, monkeypatch):
        # [FUNCTION DOC] With the ray availability probe patched to return available, a ray run
        # config is reported available (no kubectl involved).
        """A ray lane must not be gated on kubectl, helm or a cluster it never talks to."""
        monkeypatch.setattr(
            cluster_backends, "ray_availability", lambda: cluster_backends.BackendAvailability(True, "importable")
        )

        assert cluster_backends.availability_of_run(config=_run_config(ClusterBackend.RAY)).available


class TestRequireBackend:
    # [CLASS DOC] Tests of the require-backend helper, which skips or returns a namespace.
    def test_skips_with_the_reason_attached(self, monkeypatch):
        # [FUNCTION DOC] With the availability table patched so kubernetes is unavailable with a
        # reason, requiring it raises a (skip) exception whose message contains that reason.
        """A skip nobody can act on is as unhelpful as a silent pass."""
        monkeypatch.setattr(
            cluster_backends,
            "_AVAILABILITY",
            {"kubernetes": lambda: cluster_backends.BackendAvailability(False, "no cluster today")},
        )

        with pytest.raises(BaseException, match="no cluster today"):
            cluster_backends.require_backend("kubernetes")

    def test_hands_back_the_namespace_when_the_backend_is_usable(self, monkeypatch):
        # [FUNCTION DOC] With kubernetes patched available and the namespace variable set to
        # miles-e2e-mine, requiring kubernetes returns that namespace.
        """Tests install into it, so they need it and must not read the environment themselves."""
        monkeypatch.setattr(
            cluster_backends,
            "_AVAILABILITY",
            {"kubernetes": lambda: cluster_backends.BackendAvailability(True, "fine")},
        )
        monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "miles-e2e-mine")

        assert cluster_backends.require_backend("kubernetes") == "miles-e2e-mine"


class TestBothBackends:
    # [CLASS DOC] Test of the both-backends parametrizing decorator.
    @cluster_backends.both_backends
    def test_runs_once_per_backend(self, cluster_backend):
        # [FUNCTION DOC] Receives each backend name from the decorator and asserts it is ray or
        # kubernetes.
        """One test body covering both is the point; a parallel copy would only test the copy."""
        assert cluster_backend in ("ray", "kubernetes")


def _fake_kubectl(monkeypatch, respond) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Patches (reverted by monkeypatch) the tool lookup to report every
    # tool under a fixed bin directory, sets the namespace variable to mine, and replaces the
    # process runner with a fake that records each argument list and returns what the respond
    # callback returns for it. Returns the list of recorded calls, filling as the probe runs.
    calls: list[list[str]] = []

    def fake_run_process(argv, *, capture_output, check, input=None, timeout=None):
        # [FUNCTION DOC] HELPER: Fake process runner; records the argument list and returns the
        # respond callback's result.
        calls.append(argv)
        return respond(argv)

    monkeypatch.setattr(cluster_backends.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv(cluster_backends.NAMESPACE_ENV_VAR, "mine")
    monkeypatch.setattr(cluster_backends, "run_process", fake_run_process)
    return calls


def _run_config(backend: ClusterBackend, *, namespace: str = "miles-e2e") -> command_utils.ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Builds a launcher train config for the backend and namespace
    # (default miles-e2e).
    return command_utils.ExecuteTrainConfig(cluster_backend=backend, namespace=namespace)


def _completed(returncode=0, stdout="", stderr=""):
    # [FUNCTION DOC] HELPER: Builds a finished-process result with empty arguments and the given
    # return code (default 0), standard output and standard error.
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)
