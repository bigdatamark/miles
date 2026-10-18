# [MODULE DOC] Unit tests for the entry point of the colocate pairing service (a long-running
# process that discovers pods of a Helm release through the Kubernetes API). Covered: the command
# line entry rejects a config missing required fields, and the main async loop first loads the
# in-cluster Kubernetes configuration (falling back to the local kubeconfig when that is
# unavailable) and then lists the pods of the configured namespace by the release's instance label
# before waiting indefinitely on a pod watch. The Kubernetes client classes, config loaders and
# watch are replaced with fakes through monkeypatching, so no cluster is contacted.
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes_asyncio import watch as kubernetes_watch
from pydantic import ValidationError

from miles.utils.external_utils.colocate_pairing import __main__ as main_module
from miles.utils.external_utils.colocate_pairing.config import PairingConfig


def _config_json() -> str:
    # [FUNCTION DOC] HELPER: Returns the JSON text of a minimal valid pairing configuration:
    # namespace "training", release "example", trainer pool id "trainer", and one inference pool
    # "inference" whose layout has one inference cell, one trainer cell, one pod per cell of each
    # kind, 8 GPUs per node, 8 GPUs per inference pod and GPU offset 0.
    return """{
        "namespace": "training",
        "release": "example",
        "trainer_pool_id": "trainer",
        "inference_pools": [{
            "pool_id": "inference",
            "layout": {
                "num_inference_cells": 1,
                "num_trainer_cells": 1,
                "num_pods_per_inference_cell": 1,
                "num_pods_per_trainer_cell": 1,
                "num_gpus_per_node": 8,
                "num_gpus_per_inference_pod": 8,
                "gpu_offset": 0
            }
        }]
    }"""


def _config() -> PairingConfig:
    # [FUNCTION DOC] HELPER: Parses the minimal configuration text into a pairing configuration
    # object (raises a validation error if the text were invalid).
    return PairingConfig.model_validate_json(_config_json())


class TestMain:
    # [CLASS DOC] Tests of the synchronous command line entry function, which takes the argument
    # list, parses the "--config" option as JSON text into a validated configuration and then runs
    # the service.
    def test_main_rejects_a_rendered_config_missing_required_fields(self) -> None:
        # [FUNCTION DOC] Passing a config with only the namespace raises a validation error
        # whose message mentions the missing "release" field; the service is never started.
        """A rendered config missing required fields is rejected before startup."""
        with pytest.raises(ValidationError, match="release"):
            main_module.main(["--config", '{"namespace": "training"}'])


class TestRunForever:
    # [CLASS DOC] Tests of the main async coroutine that starts Kubernetes pod discovery and
    # never returns until cancelled.
    async def test_run_forever_falls_back_to_the_kubeconfig_when_not_in_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Async test. Replaces the in-cluster loader with one that records its
        # call and raises the Kubernetes config exception, the kubeconfig loader with one that
        # records its call, and the API client with a fake whose async context entry raises a
        # private sentinel error. Runs the coroutine and expects that sentinel (startup was
        # reached after configuration), then asserts the recorded calls are in-cluster first and
        # kubeconfig second.
        """An unavailable in-cluster config falls back before startup continues."""
        calls: list[str] = []

        def load_incluster_config() -> None:
            # [FUNCTION DOC] HELPER: Fake in-cluster loader: records the call and raises the
            # Kubernetes configuration exception.
            calls.append("incluster")
            raise main_module.kube_config.ConfigException()

        async def load_kube_config() -> None:
            # [FUNCTION DOC] HELPER: Fake kubeconfig loader: records the call and returns
            # nothing.
            calls.append("kubeconfig")

        class StartupReached(Exception):
            # [CLASS DOC] HELPER: Sentinel error signalling the coroutine got past the
            # configuration step and tried to open the API client.
            pass

        class FakeApiClient:
            # [CLASS DOC] HELPER: Fake API client whose async context entry raises the
            # sentinel error and whose exit does nothing.
            async def __aenter__(self) -> Any:
                # [FUNCTION DOC] HELPER: Raises the sentinel error.
                raise StartupReached

            async def __aexit__(self, *args: Any) -> None:
                # [FUNCTION DOC] HELPER: Does nothing and does not suppress exceptions.
                return None

        monkeypatch.setattr(main_module.kube_config, "load_incluster_config", load_incluster_config)
        monkeypatch.setattr(main_module.kube_config, "load_kube_config", load_kube_config)
        monkeypatch.setattr(main_module.client, "ApiClient", FakeApiClient)

        with pytest.raises(StartupReached):
            await main_module._run_forever(_config())

        assert calls == ["incluster", "kubeconfig"]

    async def test_run_forever_starts_kubernetes_discovery_and_waits_until_terminated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Async test. Installs the fake Kubernetes layer, runs the coroutine as a
        # task and waits (at most 1 second) until the fake pod listing is called. Asserts the one
        # listing call used namespace "training" and the label selector for the instance label
        # equal to release "example", and the task is still running. Then cancels the task and
        # expects the cancellation error to propagate.
        """Startup lists the configured pods before entering its terminal wait."""
        boundary = _install_kubernetes_boundary(monkeypatch)

        task = asyncio.create_task(main_module._run_forever(_config()))
        await asyncio.wait_for(boundary.discovery_started.wait(), timeout=1.0)

        assert boundary.list_calls == [
            {"namespace": "training", "label_selector": "app.kubernetes.io/instance=example"}
        ]
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _install_kubernetes_boundary(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Replaces, through monkeypatching, the service module's Kubernetes
    # pieces with fakes: the in-cluster loader (does nothing), the API client (async context
    # manager returning itself), the core API class (its namespaced pod listing records the
    # namespace and label selector keyword arguments, sets a "discovery started" event and returns
    # an empty pod list with resource version "1"), and the Kubernetes watch class (its stream
    # blocks forever on a never-completed future; close does nothing). Returns a namespace object
    # holding the list of recorded listing calls and the discovery-started event.
    boundary = SimpleNamespace(list_calls=[], discovery_started=asyncio.Event())

    class FakeApiClient:
        # [CLASS DOC] HELPER: Fake API client usable as an async context manager.
        async def __aenter__(self) -> FakeApiClient:
            # [FUNCTION DOC] HELPER: Returns itself.
            return self

        async def __aexit__(self, *args: Any) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            return None

    class FakeCoreV1Api:
        # [CLASS DOC] HELPER: Fake core API object built from an API client; only the namespaced
        # pod listing is implemented.
        def __init__(self, api_client: FakeApiClient) -> None:
            # [FUNCTION DOC] HELPER: Stores the API client.
            self.api_client = api_client

        async def list_namespaced_pod(self, *, namespace: str, label_selector: str) -> SimpleNamespace:
            # [FUNCTION DOC] HELPER: Records the namespace and label selector, signals the
            # discovery-started event, and returns an object with no pod items and metadata
            # whose resource version is "1".
            boundary.list_calls.append({"namespace": namespace, "label_selector": label_selector})
            boundary.discovery_started.set()
            return SimpleNamespace(items=[], metadata=SimpleNamespace(resource_version="1"))

    class FakeWatch:
        # [CLASS DOC] HELPER: Fake Kubernetes watch whose stream never produces an item.
        async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            # [FUNCTION DOC] HELPER: Async generator that waits forever on a future that is
            # never resolved (so it only ends through cancellation); the following bare yield
            # exists only to make the function an async generator.
            await asyncio.Future()
            yield

        async def close(self) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            return None

    monkeypatch.setattr(main_module.kube_config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(main_module.client, "ApiClient", FakeApiClient)
    monkeypatch.setattr(main_module.client, "CoreV1Api", FakeCoreV1Api)
    monkeypatch.setattr(kubernetes_watch, "Watch", FakeWatch)
    return boundary
