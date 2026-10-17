# [MODULE DOC] Worker specs, naming helpers and provider/handle factories for the training side:
# one trainer controller per trainer (a CPU-only serve worker) and one trainer pool per trainer
# (GPU workers running the backend's training actor, with cells = independent data-parallel
# replicas when independent DP is on). A trainer is one policy (actor role) or its critic. Also
# supports reaching externally deployed trainer controllers by address, and builds trainer worker
# environment variables, including memory-saver preload for offloading. Re-exports the actor and
# critic role constants imported from the Megatron config. Pool ids: trainer controller pools are
# prefixed by a shared prefix plus the trainer id; trainer pools are "trainer-engine-<trainer id>".
import os
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import (
    ACTOR_ROLE,
    CRITIC_ROLE,
    MegatronTrainerConfig,
    compute_trainer_args,
    resolve_megatron_config,
)
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import (
    TRAINER_CONTROLLER_POOL_ID_PREFIX,
    compute_cell_id,
    compute_worker_name,
    format_name_index,
)
from miles.utils.workers.types import DeployComponent, DeploymentIdentity, PlatformAccess
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.static import StaticWorkerProvider, parse_host_and_port
from miles.utils.workers.worker_spec import (
    MASTER_PORT_NAME,
    HostAndPort,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerLaunchContext,
)

TRAINER_CONTROLLER_ADDRS_FLAG = "--trainer-controller-addrs"
POOL_CATEGORY_TRAINER_ENGINE = "trainer_engine"

TRAINER_CONCURRENCY_GROUPS = {"heartbeat_status": 1, "default": 1, "fault_injector": 1, "kill_self": 1}

TRAINER_CONTROLLER_WORKER_CLASS = "miles.ray.train.group.TrainerController"

_TRAINER_ACTOR_CLASSES = {
    "megatron": "miles.backends.megatron_utils.actor.MegatronTrainRayActor",
    "fsdp": "miles.backends.fsdp_utils.actor.FSDPTrainRayActor",
    "torchtitan": "miles.backends.torchtitan_utils.actor.TorchtitanTrainRayActor",
}

_NUM_GPUS_PER_TRAINER_WORKER = 0.4


def specs_trainer_controller(args) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] One controller spec per trainer config, in config order. For each, derives the
    # trainer-specific args and sets: with-reference-model if the trainer is not a critic and either
    # the KL coefficient is non-zero or the KL loss is used; with-OPD-teacher if the trainer is not a
    # critic, on-policy distillation is on and its type is megatron.
    specs = []
    for config in compute_trainer_configs(args):
        trainer_args = compute_trainer_args(args, config)
        specs.append(
            _compute_spec_trainer_controller(
                args,
                config=config,
                with_ref=(config.role != CRITIC_ROLE) and (trainer_args.kl_coef != 0 or trainer_args.use_kl_loss),
                with_opd_teacher=(config.role != CRITIC_ROLE)
                and trainer_args.use_opd
                and trainer_args.opd_type == "megatron",
            )
        )
    return specs


def compute_trainer_configs(args) -> list[MegatronTrainerConfig]:
    # [FUNCTION DOC] The list of trainer configs (actors and critic) from the resolved Megatron config.
    return resolve_megatron_config(args).trainers


def external_trainer_controller_addrs(args, *, trainer_ids: list[str]) -> dict[str, HostAndPort] | None:
    # [FUNCTION DOC] Parse the externally deployed trainer controller addresses. Returns none when the
    # option is unset. Each entry has the form "<trainer id>=<host:port>" (split at the first equals
    # sign; an entry without one yields an empty address part). Returns a mapping from trainer id to
    # host and port. Asserts the set of ids equals the expected trainer ids and that no id repeats
    # (assertion error with the expected format message otherwise).
    if (entries := args.trainer_controller_addrs) is None:
        return None
    addrs = {prefix: parse_host_and_port(rest) for prefix, _, rest in (entry.partition("=") for entry in entries)}
    assert sorted(addrs) == sorted(trainer_ids) and len(addrs) == len(entries), (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} must name each of {trainer_ids} exactly once as "
        f"'<trainer_id>=<host:port>' (got {entries})"
    )
    return addrs


def compute_trainer_ids(args) -> list[str]:
    # [FUNCTION DOC] Trainer ids in trainer config order.
    return [config.trainer_id for config in compute_trainer_configs(args)]


def create_trainer_controller_handle(args, *, capability: BackendCapability, trainer_id: str) -> BaseWorkerHandle:
    # [FUNCTION DOC] Handle (RPC proxy) to the controller worker of a trainer, obtained from the
    # appropriate provider (static pool or external address). Does not contact the worker.
    provider = _compute_trainer_controller_provider(args, capability=capability, trainer_id=trainer_id)
    return provider.get_handle(trainer_controller_worker_name(trainer_id))


def _compute_trainer_controller_provider(
    args, *, capability: BackendCapability, trainer_id: str
) -> BaseWorkerProvider:
    # [FUNCTION DOC] HELPER: If no external trainer controller addresses are configured, the backend
    # capability's static provider for the trainer's controller pool. Otherwise a static provider over
    # the single RPC address configured for this trainer id (the controller worker class), key error if
    # the id is absent (validated by the address parser).
    pool_id = compute_trainer_controller_pool_id(trainer_id)
    if (addrs := external_trainer_controller_addrs(args, trainer_ids=compute_trainer_ids(args))) is None:
        return capability.static_worker_provider(pool_id=pool_id)
    return StaticWorkerProvider.of_rpc_addrs(
        pool_id=pool_id, addrs=[addrs[trainer_id]], worker_class=TRAINER_CONTROLLER_WORKER_CLASS
    )


def compute_trainer_controller_pool_id(trainer_id: str) -> str:
    # [FUNCTION DOC] Controller pool id: the shared controller prefix followed by the trainer id.
    return f"{TRAINER_CONTROLLER_POOL_ID_PREFIX}{trainer_id}"


def trainer_controller_worker_name(trainer_id: str) -> str:
    # [FUNCTION DOC] Worker name of a trainer's controller (single-cell pool).
    return compute_worker_name(pool_id=compute_trainer_controller_pool_id(trainer_id))


def trainer_controller_cell_id(trainer_id: str) -> str:
    # [FUNCTION DOC] Cell id of a trainer controller's only cell (index zero).
    return compute_cell_id(pool_id=compute_trainer_controller_pool_id(trainer_id), cell_index=0)


def _compute_spec_trainer_controller(
    args,
    *,
    config: MegatronTrainerConfig,
    with_ref: bool,
    with_opd_teacher: bool,
) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Controller spec for one trainer: trainer deploy component, read and delete
    # platform access, no ports or extra env, one cell with one worker using one CPU and no GPU.
    # Constructor arguments (lazy, from the launch context): a deployment identity (run uuid, deploy
    # component, deploy instance id, trainer id from args and config), the trainer id and role, the
    # reference-model and OPD teacher flags, a dynamic cell provider over the trainer's own pool, and
    # the backend's cell operations.
    trainer_id = config.trainer_id
    return ServeWorkerSpec(
        name=compute_trainer_controller_pool_id(trainer_id),
        deploy_component=DeployComponent.TRAINER,
        platform_access=PlatformAccess.READ_DELETE,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
        ),
        worker_class=TRAINER_CONTROLLER_WORKER_CLASS,
        ctor_kwargs=lambda ctx: dict(
            deployment_identity=DeploymentIdentity(
                run_uuid=args.run_uuid,
                deploy_component=args.deploy_component,
                deploy_instance_id=args.deploy_instance_id,
                trainer_id=trainer_id,
            ),
            trainer_id=trainer_id,
            role=config.role,
            with_ref=with_ref,
            with_opd_teacher=with_opd_teacher,
            cell_provider=ctx.capability.dynamic_worker_provider(pool_ids=[compute_trainer_pool_id(trainer_id)]),
            cell_operations=ctx.capability.cell_operations(),
        ),
    )


def specs_trainer(args) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] One trainer pool spec per trainer config, in order. A critic uses the critic node
    # and GPU-per-node counts and starts at placement-group slot 0; each actor uses the actor counts
    # and starts at slot (actor ordinal times GPUs per actor instance), so actors tile the group one
    # after another (critic shares slots with the first actor). Each spec is built with the trainer's
    # own derived args.
    # TODO: support different sizes after the args refactor
    actor_gpus_per_instance = args.actor_num_nodes * args.actor_num_gpus_per_node
    specs = []
    actor_index = 0
    for config in compute_trainer_configs(args):
        if config.role == CRITIC_ROLE:
            num_nodes, num_gpus_per_node, pg_slot_offset = args.critic_num_nodes, args.critic_num_gpus_per_node, 0
        else:
            num_nodes, num_gpus_per_node = args.actor_num_nodes, args.actor_num_gpus_per_node
            pg_slot_offset = actor_index * actor_gpus_per_instance
            actor_index += 1
        specs.append(
            _compute_spec_trainer(
                compute_trainer_args(args, config),
                config=config,
                num_nodes=num_nodes,
                num_gpus_per_node=num_gpus_per_node,
                pg_slot_offset=pg_slot_offset,
            )
        )
    return specs


def compute_trainer_pool_id(trainer_id: str) -> str:
    # [FUNCTION DOC] Trainer pool id "trainer-engine-<trainer id>".
    return f"trainer-engine-{trainer_id}"


def compute_trainer_num_cells(args, *, role: str) -> int:
    # [FUNCTION DOC] Number of cells (independent replicas) of a trainer pool. Takes nodes and GPUs per
    # node from the actor settings for the actor role, else from the critic settings. With independent
    # data parallelism on, total GPUs divided (integer division) by the Megatron world size excluding
    # data parallelism; otherwise exactly 1.
    num_nodes, num_gpus_per_node = (
        (args.actor_num_nodes, args.actor_num_gpus_per_node)
        if role == ACTOR_ROLE
        else (args.critic_num_nodes, args.critic_num_gpus_per_node)
    )
    total_gpus = num_nodes * num_gpus_per_node
    return (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1


def _compute_spec_trainer(
    args,
    *,
    config: MegatronTrainerConfig,
    num_nodes: int,
    num_gpus_per_node: int,
    pg_slot_offset: int,
) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Pool spec for one trainer. Cells from the cell-count helper; asserts total
    # GPUs divide evenly among cells; workers per cell = GPUs per cell. The FP8 block-scaling scales
    # setting comes from the process environment if set, else from the default helper. Scheduling:
    # each worker takes 0.4 of a GPU and 0.4 of a CPU (a fractional share, so other actors can share
    # the GPU) and one placement slot, placement group "actor" at the given slot offset. One master
    # port (static 9000, dynamic allowed). Env from the trainer env helper with the scales. Worker
    # class: the multi-LoRA Megatron actor for the megatron backend, actor role and multi-LoRA
    # enabled; otherwise the actor class for the configured backend (key error for unknown backend).
    # Constructor arguments: args, world size = GPUs per cell, rank = worker index in cell, role and
    # cell index. Concurrency groups are set only when fault tolerance is used. Metadata: role and
    # cell index.
    trainer_id = config.trainer_id
    total_gpus = num_nodes * num_gpus_per_node
    num_cells = compute_trainer_num_cells(args, role=config.role)
    assert total_gpus % num_cells == 0, f"{total_gpus=} must be divisible by {num_cells=}"
    gpus_per_cell = total_gpus // num_cells

    fp8_scales = (
        x
        if (x := os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES")) is not None
        else default_fp8_block_scaling_fp32_scales()
    )

    return ServeWorkerSpec(
        name=compute_trainer_pool_id(trainer_id),
        category=POOL_CATEGORY_TRAINER_ENGINE,
        deploy_component=DeployComponent.TRAINER,
        port_infos=[PortInfo(name=MASTER_PORT_NAME, static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda ctx: compute_trainer_env_vars(args, ctx, fp8_scales=fp8_scales),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_cpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=num_gpus_per_node,
            pg_name="actor",
            pg_slot_offset=pg_slot_offset,
        ),
        worker_class=(
            "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"
            if args.train_backend == "megatron" and config.role == ACTOR_ROLE and is_multi_lora_enabled(args)
            else _TRAINER_ACTOR_CLASSES[args.train_backend]
        ),
        ctor_kwargs=lambda ctx: dict(
            args=args,
            world_size=gpus_per_cell,
            rank=ctx.worker_in_cell_index,
            role=config.role,
            cell_index=ctx.cell_index,
        ),
        concurrency_groups=TRAINER_CONCURRENCY_GROUPS if args.use_fault_tolerance else None,
        meta=lambda ctx: dict(role=config.role, cell_index=ctx.cell_index),
    )


def compute_trainer_env_vars(args, ctx: WorkerLaunchContext, *, fp8_scales: str) -> dict[str, str]:
    # [FUNCTION DOC] Environment for one trainer worker. Base: NCCL cumem enable (process value else
    # "0", since SGLang always sets it to 0 and both must agree), NVSHMEM NCCL disabled default "1",
    # the FP8 block scaling scales, the "do not set visible devices" variables set to "1", then the
    # user's train env vars from args (which override). If a train dumper source patcher config is set,
    # adds it. If training offload is on with the megatron backend: locates the torch memory saver
    # preload library (lazy import; reads it only for the disk case), sets library preload to it and
    # enables the memory saver. For disk offload target: asserts the library contains the disk backup
    # feature marker by reading the whole binary file (assertion error telling to reinstall at the
    # pinned commit), disables CPU backup, enables disk backup, sets the chunk size in MB and a backup
    # directory under the configured disk dir named by cell and worker indices. Otherwise (CPU
    # target) enables CPU backup. Returns the mapping. Reads files when offloading to disk.
    env_vars = {
        # because sglang will always set NCCL_CUMEM_ENABLE to 0
        # we need also set it to 0 to prevent nccl error.
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
        "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": fp8_scales,
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **args.train_env_vars,
    }

    if source_patcher_config := args.dumper_source_patcher_config_train:
        env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

    if args.offload_train and args.train_backend == "megatron":
        from torch_memory_saver.utils import get_binary_path_from_package

        dynlib_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))

        env_vars["LD_PRELOAD"] = dynlib_path
        env_vars["TMS_INIT_ENABLE"] = "1"
        if args.offload_train_target == "disk":
            assert b"TMS_INIT_ENABLE_DISK_BACKUP" in Path(dynlib_path).read_bytes(), (
                f"{dynlib_path} has no disk backend; reinstall torch_memory_saver at the commit "
                f"docker/Dockerfile pins."
            )
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "0"
            env_vars["TMS_INIT_ENABLE_DISK_BACKUP"] = "1"
            env_vars["TMS_DISK_BACKUP_CHUNK_MB"] = str(args.offload_train_disk_chunk_mb)
            env_vars["TMS_DISK_BACKUP_DIR"] = os.path.join(
                args.offload_train_disk_dir,
                f"cell{format_name_index(ctx.cell_index)}_rank{format_name_index(ctx.worker_in_cell_index)}",
            )
        else:
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    return env_vars
