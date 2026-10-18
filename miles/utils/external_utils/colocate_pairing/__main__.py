# [MODULE DOC] Entry point of the colocate pairing controller process (run as a module). Parses a
# single required argument, a json-encoded pairing config rendered by the deployment chart, then
# runs forever: loads in-cluster Kubernetes credentials (falling back to the local kube config),
# builds the controller, a pod reflector limited to the release's pods in the configured
# namespace, and a reconcile loop (resync every 300 seconds) keyed by the trainer pod each pod is
# paired with, then blocks until the process is killed.
from __future__ import annotations

import argparse
import asyncio
import logging

from kubernetes_asyncio import client
from kubernetes_asyncio import config as kube_config

from miles.utils.external_utils.colocate_pairing.config import PairingConfig
from miles.utils.external_utils.colocate_pairing.controller import PairingController
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

logger = logging.getLogger(__name__)

_RESYNC_PERIOD_SECONDS = 300.0


def main(argv: list[str] | None = None) -> int:
    # [FUNCTION DOC] Parses the arguments (defaults to the process arguments) with a required
    # config option, configures root logging at info level with a timestamped format, validates the
    # json into a pairing config (invalid input raises a validation error), and runs the async
    # main loop to completion (normally never). Returns 0.
    # KNOWN ISSUE: the argument parser description is taken from the module docstring, but this
    # module has none, so the description is empty.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A PairingConfig as json, rendered by the chart")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = PairingConfig.model_validate_json(args.config)
    asyncio.run(_run_forever(config))
    return 0


async def _run_forever(config: PairingConfig) -> None:
    # [FUNCTION DOC] HELPER: Loads in-cluster Kubernetes configuration, or, if not running in a
    # cluster, the local kube config file. Opens an API client, creates the controller, a pod
    # reflector selecting pods labelled with the instance label equal to the release name in the
    # namespace, and a reconcile loop that reads the reflector's watch stream, reconciles by pair
    # key (the controller's key mapping) and resyncs every 300 seconds; hands the loop to the
    # controller; enters the loop, logs what is watched, and waits on an event that is never set.
    try:
        kube_config.load_incluster_config()
    except kube_config.ConfigException:
        await kube_config.load_kube_config()

    async with client.ApiClient() as api_client:
        core_v1 = client.CoreV1Api(api_client)
        controller = PairingController(config=config, core_v1=core_v1)
        reflector = KubernetesReflector(
            kube_client=KubernetesAsyncioPodApi(core_v1_api=core_v1),
            namespace=config.namespace,
            label_selector=f"{INSTANCE_LABEL}={config.release}",
        )
        loop = ReconcileLoop(
            source=reflector.watch,
            reconcile=controller.reconcile,
            key_map=controller.key_of,
            resync_period=_RESYNC_PERIOD_SECONDS,
        )
        controller.set_loop(loop)
        async with loop:
            logger.info("Pairing controller watching %s in %s", config.release, config.namespace)
            await asyncio.Event().wait()


if __name__ == "__main__":
    raise SystemExit(main())
