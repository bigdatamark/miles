# [MODULE DOC]
# Command-line entrypoint for a deployed component that has no orchestration script: it launches
# the worker manager so the component's workers run, optionally serves fault tolerance for the
# trainer cells, and then stays alive until the process is torn down.
import asyncio
import logging

# TODO: this entrypoint reaches into miles.ray; fix the layering later
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.train import compute_trainer_ids, create_trainer_controller_handle
from miles.ray.wiring import get_backend_capability, launch_worker_manager
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.workers.types import DeployComponent

logger = logging.getLogger(__name__)


def main() -> None:
    # [FUNCTION DOC]
    # Parse the command-line arguments and run the serving coroutine to completion on a new event
    # loop. Normally never returns, since serving waits forever.
    asyncio.run(_serve_deployed_workers(parse_args()))


async def _serve_deployed_workers(args) -> None:
    # [FUNCTION DOC] HELPER: Main flow. Configure logging with a process identity named main; read
    # the deployed component from the arguments and assert it does not carry an orchestration
    # script (such components must be started through their driver script). Launch the worker
    # manager, log the names of the specs of this run, start the fault-tolerance API if applicable,
    # then block forever on an event that is never set.
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    component = DeployComponent(args.deploy_component)
    assert not component.deploys_orchestration_script(), (
        f"this entrypoint installs the workers of a deployment that carries no orchestration script, and "
        f"--deploy-component {component.value} carries one: launch it through its driver script instead"
    )

    _worker_manager = launch_worker_manager(args)
    logger.info(f"Deployed the {component.value} workers of this run: {[spec.name for spec in compute_specs(args)]}")
    _maybe_serve_fault_tolerance(args, component=component)
    logger.info(
        "This deployment carries no orchestration script, so it has no training to finish and stays up until it is "
        "torn down"
    )

    await asyncio.Event().wait()


def _maybe_serve_fault_tolerance(args, *, component: DeployComponent) -> None:
    # [FUNCTION DOC] HELPER: Does nothing unless an API server port is configured and the deployed
    # component selects the trainer. Otherwise builds the backend capability, then starts the API
    # server with one trainer controller handle per trainer id (computed from the arguments), no
    # inference controller, the configured port and fault-tolerance components, and the cell
    # operations of the capability. Then starts the mini fault-tolerance controller if configured
    # and logs the port.
    if not args.api_server_port or not component.selects(DeployComponent.TRAINER):
        return

    capability = get_backend_capability(args)
    start_api_server(
        args=args,
        trainer_models={
            trainer_id: create_trainer_controller_handle(args, capability=capability, trainer_id=trainer_id)
            for trainer_id in compute_trainer_ids(args)
        },
        inference_controller=None,
        port=args.api_server_port,
        ft_components=args.ft_components,
        cell_operations=capability.cell_operations(),
    )
    maybe_start_mini_ft_controller(args)
    logger.info(f"Serving the fault tolerance of this deployment's own cells on port {args.api_server_port}")


if __name__ == "__main__":
    main()
