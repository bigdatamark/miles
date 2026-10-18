# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Client construction for the kind suite: builds an asynchronous Kubernetes API
# client from a kubeconfig file.

from __future__ import annotations

from pathlib import Path

from kubernetes_asyncio import client as kubernetes_client
from kubernetes_asyncio import config as kubernetes_config


async def build_kubeconfig_api_client(*, kubeconfig: Path) -> kubernetes_client.ApiClient:
    # [FUNCTION DOC] HELPER: Loads the kubeconfig file (current context) into a fresh configuration
    # object and returns an API client using it. The caller must close the client.
    configuration = kubernetes_client.Configuration()
    await kubernetes_config.load_kube_config(config_file=str(kubeconfig), client_configuration=configuration)
    return kubernetes_client.ApiClient(configuration=configuration)
