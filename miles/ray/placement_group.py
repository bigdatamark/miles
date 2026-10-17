# [MODULE DOC] Ray placement-group creation and the orchestration glue that brings up trainers and
# rollout components for a run.
# Responsibilities: (1) reserve one Ray placement group of GPU bundles and reorder its bundles by node
# and GPU id so that logically adjacent ranks land on adjacent physical GPUs; (2) compute how many
# GPUs the trainer / rollout sides need under the debug, external-rollout and colocate modes;
# (3) create worker handles for trainers, take over or wait for externally deployed trainer
# controllers, restore trainer state and decide the starting rollout id; (4) create the rollout
# executor and inference controller handles and derive the rollout count from epochs; (5) drive one
# weight update from trainer to inference engines and optionally log engine weight checksums;
# (6) optionally start the fault-tolerance API server.
# Several functions are marked "move when reorganizing files": they are orchestration helpers that
# do not conceptually belong with placement groups.
import asyncio
import logging
import socket
from typing import NamedTuple

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.backends.megatron_utils.megatron_config import MegatronTrainerConfig, compute_trainer_args
from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.specs.inference import (
    SESSION_SERVER_POOL_ID,
    compute_router_providers,
    create_inference_controller_handle,
)
from miles.ray.specs.rollout import create_rollout_executor_handle
from miles.ray.specs.train import (
    ACTOR_ROLE,
    CRITIC_ROLE,
    TRAINER_CONTROLLER_ADDRS_FLAG,
    compute_trainer_configs,
    create_trainer_controller_handle,
    external_trainer_controller_addrs,
)
from miles.ray.wiring import get_backend_capability
from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.hot_restart import (
    init_or_reset_inference_controller,
    trainer_init_or_load_state,
    wait_trainers_idle,
    wait_until_worker_not_initialized,
)
from miles.utils.test_utils.ft_test_actions import FTTestActionOrchestrationExecutor
from miles.utils.workers.types import DeployComponent, DeploymentIdentity
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.static import wait_static_addrs_ready

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    # [CLASS DOC] HELPER: Short-lived Ray actor that reserves exactly one GPU, used only to discover
    # which node and physical GPU a placement-group bundle landed on. It is killed right after use.

    def get_ip_and_gpu_id(self):
        # [FUNCTION DOC] HELPER: Returns a pair of the node IP address of the hosting node and the
        # first GPU id assigned to this actor by Ray. Takes no inputs; no side effects.
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    # [FUNCTION DOC] HELPER: Sort key for bundle info triples of (bundle index, node identifier, gpu id).
    # Returns a pair of (list of integers describing the node, gpu id) so that bundles order by node
    # first and GPU id second. Node description: if the identifier splits on dots into integers it is
    # treated as an IPv4 address; otherwise it is resolved as a hostname to an IP address and parsed
    # the same way; if resolution fails (resolver error or non-string identifier), each character of
    # the identifier is converted to its character code, giving a stable ordering. Note a dotted
    # identifier with non-integer parts that resolves is used via its resolved address. The bundle
    # index is ignored. May perform a DNS lookup as a side effect.
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


class PlacementGroupInfo(NamedTuple):
    # [CLASS DOC] Immutable record describing a placement group slice: the Ray placement group (or
    # nothing when zero GPUs were requested), the bundle indices in reordered (node, GPU) order, and
    # the physical GPU id of each bundle in the same order. Position i in both lists is logical
    # index i; consumers use the bundle index to schedule an actor and the GPU id for device mapping.
    pg: PlacementGroup
    pg_reordered_bundle_indices: list[int]
    pg_reordered_gpu_ids: list[int]


def _create_placement_group(num_gpus) -> PlacementGroupInfo:
    # [FUNCTION DOC] HELPER: Reserve num_gpus bundles of one GPU and one CPU each using the pack
    # strategy, block until the group is ready, then spawn one info actor per bundle to learn each
    # bundle's node IP and GPU id, and kill the actors afterwards. Bundles are sorted with the sort
    # key helper (node, then GPU id) and the permutation is returned. Every reordered bundle is
    # logged at info level. Returns a placement group info; when num_gpus is zero returns a plain
    # triple of no group and two empty lists (equivalent in shape but not a named tuple instance).
    # Side effects: Ray resource reservation, actor creation and kill, blocking waits.
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return PlacementGroupInfo(pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids)


def _get_placement_group_layout(args) -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Decide the total GPU count of the placement group and the offset at which
    # the rollout slice begins (equal to the number of trainer GPUs reserved at the front).
    # Returns a pair (total GPUs, rollout offset). Steps: trainer GPUs are counted only if the
    # deploy component selects the trainer, else zero. Debug train-only mode: total is trainer GPUs
    # plus eval GPUs (eval GPUs only if the deploy component selects inference), offset is trainer
    # GPUs. Otherwise rollout GPUs = (rollout GPUs, treating unset as zero, plus eval GPUs) only if
    # inference is selected, else zero. External rollout: nothing is reserved for rollout, so
    # (0, 0) in debug rollout-only mode, else (trainer, trainer). Debug rollout-only: (rollout, 0).
    # Colocate: the larger of trainer and rollout GPUs with offset zero (they share GPUs).
    # Default disaggregated: (trainer + rollout, trainer).
    selector = DeployComponent(args.deploy_component)
    trainer_num_gpus = _compute_trainer_num_gpus(args) if selector.selects(DeployComponent.TRAINER) else 0
    if args.debug_train_only:
        eval_num_gpus = args.eval_num_gpus if selector.selects(DeployComponent.INFERENCE) else 0
        return trainer_num_gpus + eval_num_gpus, trainer_num_gpus

    rollout_num_gpus = (
        (args.rollout_num_gpus or 0) + args.eval_num_gpus if selector.selects(DeployComponent.INFERENCE) else 0
    )
    if args.rollout_external:
        return (0, 0) if args.debug_rollout_only else (trainer_num_gpus, trainer_num_gpus)
    if args.debug_rollout_only:
        return rollout_num_gpus, 0
    if args.colocate:
        return max(trainer_num_gpus, rollout_num_gpus), 0
    return trainer_num_gpus + rollout_num_gpus, trainer_num_gpus


def _compute_trainer_num_gpus(args) -> int:
    # [FUNCTION DOC] HELPER: Number of GPUs all actor-role trainers need: the count of actor-role
    # trainer configs (one per policy) times actor nodes times GPUs per actor node. Critic trainers
    # are not counted (they share the actor's GPUs).
    num_policies = len([config for config in compute_trainer_configs(args) if config.role == ACTOR_ROLE])
    return args.actor_num_nodes * args.actor_num_gpus_per_node * num_policies


def create_placement_groups(args) -> dict[str, PlacementGroupInfo]:
    # [FUNCTION DOC] Create a single placement group sized by the layout helper and split it into
    # views. Returns a mapping with key actor (whole reordered bundle list) and key rollout (the
    # reordered bundle indices and GPU ids from the rollout offset onward), both sharing the same
    # placement group object; when a critic is used, key critic aliases the actor entry. With zero
    # GPUs the group is absent and lists are empty. Side effects: logs and reserves Ray resources.
    """Create placement groups for actor and rollout engines."""

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    ans = {
        "actor": PlacementGroupInfo(pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": PlacementGroupInfo(pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }
    if args.use_critic:
        ans["critic"] = ans["actor"]
    return ans


class TrainerInfo(NamedTuple):
    # [CLASS DOC] Immutable record for one initialized trainer: its worker handle, the rollout id
    # restored from checkpoint or state, and the rollout id training will actually start from
    # (which may be overridden by the user-specified start rollout id).
    handle: BaseWorkerHandle
    restored_rollout_id: int
    start_rollout_id: int


# TODO: move (when reorganizing files)
def create_trainer_handles(args, *, trainer_configs: list[MegatronTrainerConfig]) -> dict[str, BaseWorkerHandle]:
    # [FUNCTION DOC] Build one trainer controller handle per trainer config using the backend
    # capability selected from args. Returns a mapping from trainer id to worker handle. Creating a
    # handle does not itself initialize the trainer.
    capability = get_backend_capability(args)
    return {
        config.trainer_id: create_trainer_controller_handle(args, capability=capability, trainer_id=config.trainer_id)
        for config in trainer_configs
    }


# TODO: move (when reorganizing files)
async def take_over_trainers(args, *, handles: dict[str, BaseWorkerHandle]) -> bool:
    # [FUNCTION DOC] Wait for any externally deployed trainer controllers to be ready and belong to
    # this run, then wait until all trainers are idle. Returns whether the trainers were already
    # resumed (initialized by an earlier orchestrator, i.e. hot restart). If resumed but no
    # checkpoint tracker exists at the requested load path, the event logger's checkpoint state is
    # discarded as a side effect.
    await wait_external_trainers(args, handles=handles)
    resumed = await wait_trainers_idle(handles)

    if resumed and not _trainer_has_checkpoint(args):
        event_logger_checkpoint.discard(args)

    return resumed


def _trainer_has_checkpoint(args) -> bool:
    # [FUNCTION DOC] HELPER: True if a checkpoint tracker file at the requested load path records an
    # iteration. Asserts (raises assertion error) when a multi-policy megatron config is present,
    # since the base load path then has no tracker.
    assert args.megatron_config is None, "a multi policy run's base --load holds no tracker to read"
    return read_checkpoint_tracker_iteration(args.requested_load) is not None


# TODO: move (when reorganizing files)
async def create_training_model(args, *, handle: BaseWorkerHandle, trainer_id: str, resumed: bool) -> TrainerInfo:
    # [FUNCTION DOC] Initialize one trainer (or load its existing state when resumed) and return a
    # trainer info. The init call yields one restored rollout id per worker; all must be equal
    # (assertion error otherwise) and the single value is the restored rollout id. The start
    # rollout id is the restored one unless args specifies a start rollout id, which wins (an info
    # message is logged when it differs). Does not mutate args.
    restored_rollout_ids = await trainer_init_or_load_state(handle, args, trainer_id=trainer_id, resumed=resumed)
    assert len(set(restored_rollout_ids)) == 1, f"trainer {trainer_id!r} restored {restored_rollout_ids}"
    [restored_rollout_id] = set(restored_rollout_ids)

    if (x := args.start_rollout_id) is None:
        start_rollout_id = restored_rollout_id
    else:
        if x != restored_rollout_id:
            logger.info(
                f"trainer {trainer_id!r} restored rollout {restored_rollout_id}, and --start-rollout-id {x} was "
                f"asked for, so it starts at {x}"
            )
        start_rollout_id = x

    return TrainerInfo(handle=handle, restored_rollout_id=restored_rollout_id, start_rollout_id=start_rollout_id)


# TODO: move (when reorganizing files)
async def create_training_models(
    args, rollout_executor: BaseWorkerHandle
) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    # [FUNCTION DOC] Bring up all trainers and wire them to the rollout executor. Steps: compute
    # trainer configs, create handles, take over trainers (yielding the resumed flag); initialize
    # the single actor-role trainer (exactly one required) with its derived args; if a critic is
    # used, initialize exactly one critic-role trainer and assert it restored to the same rollout id
    # as the actor, otherwise assert there are no critic configs. Then write the actor's start
    # rollout id back into args (mutation), push the actor's training parallel config to the rollout
    # executor, and ask the executor to load state for the rollout before the start (start minus
    # one). Returns a pair of (actor handle, critic handle or nothing).
    trainer_configs = compute_trainer_configs(args)
    handles = create_trainer_handles(args, trainer_configs=trainer_configs)
    resumed = await take_over_trainers(args, handles=handles)

    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    actor_info = await create_training_model(
        compute_trainer_args(args, actor_config),
        handle=handles[actor_config.trainer_id],
        trainer_id=actor_config.trainer_id,
        resumed=resumed,
    )

    critic_configs = [config for config in trainer_configs if config.role == CRITIC_ROLE]
    critic_info = None
    if args.use_critic:
        [critic_config] = critic_configs
        critic_info = await create_training_model(
            compute_trainer_args(args, critic_config),
            handle=handles[critic_config.trainer_id],
            trainer_id=critic_config.trainer_id,
            resumed=resumed,
        )
        assert critic_info.restored_rollout_id == actor_info.restored_rollout_id, (
            f"the actor restored to rollout {actor_info.restored_rollout_id} but its critic to "
            f"{critic_info.restored_rollout_id}"
        )
    else:
        assert (
            not critic_configs
        ), f"a run without --use-critic needs no critic, but the trainer configs are {trainer_configs}"

    args.start_rollout_id = actor_info.start_rollout_id

    await rollout_executor.set_train_parallel_config(await actor_info.handle.get_train_parallel_config())
    await rollout_executor.load(args.start_rollout_id - 1)

    return actor_info.handle, critic_info.handle if critic_info is not None else None


# TODO: move (when reorganizing files)
async def wait_external_trainers(args, *, handles: dict[str, BaseWorkerHandle]) -> None:
    # [FUNCTION DOC] If no external trainer controller addresses are configured, do nothing.
    # Otherwise resolve the address of each trainer id, wait until the static addresses are ready,
    # fetch every handle's deployment identity concurrently, and verify each identity with the
    # identity assertion helper (raising assertion error on mismatch). Logs the addresses.
    """Wait for every independently deployed trainer controller, and refuse one that another run deployed."""
    if args.trainer_controller_addrs is None:
        return

    addrs = external_trainer_controller_addrs(args, trainer_ids=list(handles))
    logger.info(f"Waiting for the independently deployed trainer controllers at {addrs}")
    await wait_static_addrs_ready(addrs.values())

    identities = await asyncio.gather(*[handle.get_deployment_identity() for handle in handles.values()])
    for trainer_id, identity in zip(handles, identities, strict=True):
        _assert_external_trainer_in_run(identity, args=args, trainer_id=trainer_id)


def _assert_external_trainer_in_run(identity: DeploymentIdentity, *, args, trainer_id: str | None = None) -> None:
    # [FUNCTION DOC] HELPER: Validate that an externally reached trainer deployment identity belongs
    # to this launch. Raises assertion error (with explanatory message) unless: the run uuid equals
    # this run's; the deploy component is the trainer-only component (an all-in-one release would
    # run its own orchestration script); the deploy instance id is unset or equals the trainer id;
    # and the identity's trainer id equals the expected trainer id.
    assert identity.run_uuid == args.run_uuid, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, but this launch drives run {args.run_uuid}: every deployment a split run reaches has "
        f"to be a deployment of that same run, or its weight updates and its rollout samples belong to different runs"
    )
    assert identity.deploy_component == DeployComponent.TRAINER.value, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, and only a deployment that carries nothing but the trainer is reached by address: "
        f"an {DeployComponent.ALL.value} release of this run runs an orchestration script of its own, so both "
        f"scripts would drive the same trainer"
    )
    assert identity.deploy_instance_id is None or identity.deploy_instance_id == trainer_id, (
        f"trainer {trainer_id!r} answers as deployment {identity.deploy_instance_id!r}; "
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} entries are keyed by trainer id"
    )
    assert (
        identity.trainer_id == trainer_id
    ), f"{TRAINER_CONTROLLER_ADDRS_FLAG} keys trainer {identity.trainer_id!r} under {trainer_id!r}"


# TODO: move (when reorganizing files)
async def update_weights(
    args,
    actor_model,
    rollout_executor,
    inference_controller: BaseWorkerHandle,
    *,
    rollout_id: int | None = None,
    trainer_model_id: str | None = None,
) -> None:
    # [FUNCTION DOC] Push one set of trained weights to the inference engines. Steps: build the
    # fault-tolerance test action executor from args; if a rollout id is given run its after-step
    # actions; ask the inference controller to start the weight update, which returns the updatable
    # engines info; have the actor model update weights into those engines (returns a weight
    # version or nothing); on any failure (including cancellation) abort the update on the
    # controller and re-raise; otherwise end the update passing the snapshot mapping from cell id to
    # hashes. Then optionally log inference engine weight checksums and, if a weight version was
    # returned, record it on the rollout executor for the given trainer model id.
    orchestration_executor = FTTestActionOrchestrationExecutor.from_args(args, trainer_model_id=trainer_model_id)
    if rollout_id is not None:
        await orchestration_executor.run_after_step(rollout_id=rollout_id)

    info: UpdatableEngines = await inference_controller.start_update_weights(model_id=trainer_model_id)
    try:
        weight_version = await actor_model.update_weights(info=info, rollout_id=rollout_id)
    except BaseException:
        await inference_controller.abort_update_weights()
        raise
    await inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

    await _maybe_log_inference_engine_weight_checksums(
        args, inference_controller=inference_controller, rollout_id=rollout_id, trainer_model_id=trainer_model_id
    )

    if weight_version is not None:
        await rollout_executor.set_weight_version(weight_version, trainer_model_id=trainer_model_id)


async def _maybe_log_inference_engine_weight_checksums(
    args, *, inference_controller: BaseWorkerHandle, rollout_id: int | None, trainer_model_id: str | None
) -> None:
    # [FUNCTION DOC] HELPER: Ask the inference controller for weight checksums and record them in
    # the event log. Does nothing if the event logger is not initialized, in debug train-only or
    # rollout-only mode, or if the check returns empty. Otherwise flattens the per-engine checksums
    # and logs an engine weight checksum event carrying the rollout id (if none is given, the one
    # before the start rollout id), the trainer model id and the checksums.
    if not is_event_logger_initialized():
        return
    if args.debug_train_only or args.debug_rollout_only:
        return

    check_weights_result = await inference_controller.check_weights(action="checksum", model_id=trainer_model_id)
    if not check_weights_result:
        return
    engine_checksums = flatten_inference_engine_checksums(check_weights_result)
    get_event_logger().log(
        InferenceEngineWeightChecksumEvent,
        dict(
            rollout_id=args.start_rollout_id - 1 if rollout_id is None else rollout_id,
            trainer_model_id=trainer_model_id,
            engine_checksums=engine_checksums,
        ),
    )


# TODO: move (when reorganizing files)
def maybe_start_api_server(
    args, *, trainer_models: dict[str, BaseWorkerHandle], inference_controller: BaseWorkerHandle
) -> None:
    # [FUNCTION DOC] Start the fault-tolerance API server in-process if a port is configured (a
    # falsy port means disabled and nothing happens). Passes the trainer handles, inference
    # controller, host, port, fault-tolerance components and the backend's cell operations.
    if not args.api_server_port:
        return

    start_api_server(
        args=args,
        trainer_models=trainer_models,
        inference_controller=inference_controller,
        host=args.api_server_host,
        port=args.api_server_port,
        ft_components=args.ft_components,
        cell_operations=get_backend_capability(args).cell_operations(),
    )


class RolloutComponents(NamedTuple):
    # [CLASS DOC] Immutable bundle of the rollout side handles: the inference controller, the
    # rollout executor, and the number of rollouts per epoch (only set when it had to be computed
    # because the total rollout count was not given).
    inference_controller: BaseWorkerHandle
    rollout_executor: BaseWorkerHandle
    num_rollout_per_epoch: int | None


# TODO: move (when reorganizing files)
async def create_rollout_components(args) -> RolloutComponents:
    # [FUNCTION DOC] Create and initialize the rollout executor and inference controller. Steps:
    # unless running train-only without eval GPUs, resolve router addresses from the router
    # providers and wait for the session server (a static worker provider for the session pool only
    # if the session server is enabled). Create the rollout executor handle and wait until it is not
    # yet initialized; create the inference controller handle and init or reset it; initialize the
    # executor. If args has no total rollout count, fetch rollouts per epoch from the executor and
    # set the total in args (mutation) to that times the epoch count, asserting it is positive. If
    # the controller reports eval fleet info, forward it to the executor. Returns the components.
    capability = get_backend_capability(args)

    if not args.debug_train_only or args.eval_num_gpus > 0:
        await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))

        session_server_provider = (
            capability.static_worker_provider(pool_id=SESSION_SERVER_POOL_ID) if args.use_session_server else None
        )
        await wait_session_server_ready(args, provider=session_server_provider)

    rollout_executor = create_rollout_executor_handle(capability=capability)
    await wait_until_worker_not_initialized(rollout_executor)

    inference_controller = create_inference_controller_handle(capability=capability)
    await init_or_reset_inference_controller(inference_controller, args=args)

    await rollout_executor.init()

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = await rollout_executor.get_num_rollout_per_epoch()
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if (eval_fleet_info := await inference_controller.get_eval_fleet_info()) is not None:
        await rollout_executor.set_eval_fleet_info(eval_fleet_info)

    return RolloutComponents(
        inference_controller=inference_controller,
        rollout_executor=rollout_executor,
        num_rollout_per_epoch=num_rollout_per_epoch,
    )
