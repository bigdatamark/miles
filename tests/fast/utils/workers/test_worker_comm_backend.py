# [MODULE DOC] Tests for choosing the worker communication backend (ray actor calls or rpc
# server calls). resolve_worker_comm_backend maps the cluster backend plus an optional requested
# string to a WorkerCommBackend: unset means ray for a ray cluster and rpc for kubernetes;
# explicit ray or rpc is accepted on ray; kubernetes accepts only rpc (asking for ray fails an
# assertion naming the worker-comm-backend flag); an unknown name raises a value error. The last
# class checks that the public argument validation stores the resolved value on the parsed
# arguments (also for the kubernetes cluster backend) and fails for the impossible combination.
from __future__ import annotations

import argparse

import pytest

from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend, resolve_worker_comm_backend


class TestTheAutomaticChoice:
    # [CLASS DOC] Tests of the default when nothing is requested.
    def test_ray_keeps_talking_over_ray_until_the_default_flips(self):
        # [FUNCTION DOC] Ray cluster backend with no request resolves to the ray communication
        # backend.
        """The flag exists so both modes coexist; today an unset flag must change nothing for ray users."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested=None)

        assert chosen == WorkerCommBackend.RAY

    def test_kubernetes_talks_over_rpc(self):
        # [FUNCTION DOC] Kubernetes cluster backend with no request resolves to rpc.
        """A pod is not an actor, so the only way to call it is the server it serves."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested=None)

        assert chosen == WorkerCommBackend.RPC


class TestTheExplicitChoice:
    # [CLASS DOC] Tests of explicit requests per cluster backend.
    @pytest.mark.parametrize("requested", ["ray", "rpc"])
    def test_ray_accepts_both_modes(self, requested: str):
        # [FUNCTION DOC] On a ray cluster each of "ray" and "rpc" resolves to the matching backend
        # value.
        """Ray is where the two modes coexist, which is what makes a gradual switch possible."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested=requested)

        assert chosen == WorkerCommBackend(requested)

    def test_kubernetes_accepts_rpc(self):
        # [FUNCTION DOC] On kubernetes an explicit "rpc" resolves to rpc.
        """Naming the backend that is already in use must not be an error."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested="rpc")

        assert chosen == WorkerCommBackend.RPC

    def test_kubernetes_refuses_ray_communication(self):
        # [FUNCTION DOC] On kubernetes an explicit "ray" raises an assertion error mentioning the
        # worker-comm-backend flag name.
        """There is no actor to call, so accepting the flag would fail much later and far less clearly."""
        with pytest.raises(AssertionError, match="worker-comm-backend"):
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested="ray")

    def test_an_unknown_backend_is_rejected(self):
        # [FUNCTION DOC] On a ray cluster the name "grpc" raises a value error.
        """A typo must not silently fall back to the default."""
        with pytest.raises(ValueError):
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested="grpc")


class TestTheChoiceThatIsStored:
    # [CLASS DOC] Tests that argument validation stores the resolved backend on the arguments.
    @pytest.mark.parametrize("requested, expected", [([], "ray"), (["--worker-comm-backend", "rpc"], "rpc")])
    def test_validation_leaves_the_resolved_wire_on_the_arguments(self, requested: list[str], expected: str):
        # [FUNCTION DOC] Builds a parser with the extra-arguments provider, parses the optional flag
        # plus rollout batch size 64 and one rollout, runs the validation, and asserts the stored
        # worker comm backend equals ray (flag absent) or rpc (flag given).
        """The wire is resolved once, so everything downstream reads a value rather than resolving again."""
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args([*requested, "--rollout-batch-size", "64", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.worker_comm_backend == expected

    @pytest.mark.parametrize("requested, expected", [([], "rpc"), (["--worker-comm-backend", "rpc"], "rpc")])
    def test_kubernetes_is_resolved_through_the_public_validation_too(self, requested: list[str], expected: str):
        # [FUNCTION DOC] With the cluster backend flag set to kubernetes, validated arguments store
        # rpc whether or not rpc was requested explicitly.
        """Every downstream reader takes the wire off args, so the k8s path must be resolved there as well."""
        args = _validated_args([*requested, "--cluster-backend", "kubernetes"])

        assert args.worker_comm_backend == expected

    def test_asking_kubernetes_for_ray_fails_the_whole_validation(self):
        # [FUNCTION DOC] Kubernetes cluster backend with requested ray makes validation raise an
        # assertion error mentioning the worker-comm-backend flag.
        """A combination that cannot work must stop the run at argument time, not at the first call."""
        with pytest.raises(AssertionError, match="worker-comm-backend"):
            _validated_args(["--cluster-backend", "kubernetes", "--worker-comm-backend", "ray"])


def _validated_args(argv: list[str]) -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Parses argv plus rollout batch size 64 and one rollout with a parser
    # carrying the extra arguments, runs the validation (which may raise) and returns the
    # namespace.
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args([*argv, "--rollout-batch-size", "64", "--num-rollout", "1"])
    miles_validate_args(args)
    return args
