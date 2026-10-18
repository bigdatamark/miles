# [MODULE DOC]
# HELPER: Shared utilities for deploy end-to-end tests that install runs on a Kubernetes cluster
# as helm releases. Provides: a wrapper that turns a CI scenario callable into a test entry point
# which first asserts the cluster can deploy runs; the cluster checks (only the Kubernetes
# backend is supported and a namespace must be configured, then the backend factory must succeed
# for the run); and a comparison of the baseline and target sides of a deterministic run that
# additionally checks the inference engine count recorded on each side.
from collections.abc import Callable

from tests.e2e.ft.conftest_ft import comparisons
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE
from tests.fast.cluster_backends import RUN_NAMESPACE_ENV_VAR, compute_missing_namespace_reason, create_backend_for_run

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.comparisons.inference_engine_checksums import assert_engine_count
from miles.utils.workers.types import ClusterBackend


# ============================ the cluster to run on ===========================


def run_on_cluster(run_ci: Callable[[str | None], None]) -> Callable[[], None]:
    # [FUNCTION DOC] HELPER: Wrap a scenario callable (taking an optional string argument) into a
    # zero-argument callable. Returns the wrapper.
    def run_ci_on_deployable_cluster() -> None:
        # [FUNCTION DOC] HELPER: Assert, using the default execute-train configuration, that the
        # cluster can deploy runs (see the cluster check below), then call the wrapped scenario
        # with no argument (nothing). Raises assertion errors from the check.
        assert_cluster_can_deploy_runs(command_utils.default_config())
        run_ci(None)

    return run_ci_on_deployable_cluster


def assert_cluster_can_deploy_runs(config: command_utils.ExecuteTrainConfig) -> None:
    # [FUNCTION DOC] HELPER: Assert the configuration is deployable: the unconfigured-reason check
    # must return nothing (else the assertion fails with that reason), then call the run backend
    # factory for the configuration (which may raise if the backend cannot be created; the
    # created backend is discarded).
    assert (reason := _compute_unconfigured_reason(config)) is None, reason

    create_backend_for_run(config)


def _compute_unconfigured_reason(config: command_utils.ExecuteTrainConfig) -> str | None:
    # [FUNCTION DOC] HELPER: Return a message explaining why the environment cannot host these
    # tests, or nothing. If the cluster backend is not Kubernetes, the message says helm releases
    # require it and names the declared backend. Else if no namespace is configured, return the
    # missing-namespace reason for the run-namespace environment variable. Else nothing.
    if (backend := config.cluster_backend) is not ClusterBackend.KUBERNETES:
        return (
            f"these tests install a run as helm releases, which only the {ClusterBackend.KUBERNETES.value} "
            f"backend does, and this environment declares the {backend.value} backend"
        )
    if not config.namespace:
        return compute_missing_namespace_reason(RUN_NAMESPACE_ENV_VAR)
    return None


# =========================== comparing the two sides ==========================


def compare_deterministic_sides(
    *,
    baseline_dir: str,
    target_dir: str,
    expected_engine_count: int,
    min_trained_rollouts: int,
    exclude_keys: list[str] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Compare the dump directories of the baseline and target sides of a
    # deterministic run by delegating to the fault-tolerance comparison helper (with the minimum
    # trained rollouts and optional excluded keys), then, for each side, assert the inference
    # engine count recorded in that side's dump directory equals the expected count. The expected
    # engine count is not passed to the delegate. Raises assertion errors.
    comparisons.compare_deterministic_sides(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        min_trained_rollouts=min_trained_rollouts,
        exclude_keys=exclude_keys,
    )

    for side, side_dir in ((BASELINE_SIDE, baseline_dir), (TARGET_SIDE, target_dir)):
        assert_engine_count(side=side, dump_dir=side_dir, expected=expected_engine_count)
