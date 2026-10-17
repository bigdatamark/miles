# [MODULE DOC] Worker specs, naming helpers and provider factories for the inference side of a run:
# the inference controller, optional registration reporter, per-model routers, the session server
# pool, and the SGLang inference engine pools (one pool per model and server group). Also builds
# the launch commands (router, session server, engine) and the environment variables for engines.
# Pool ids: inference-controller, session-server, inference-registration-reporter,
# inference-router-<model index>, inference-engine-<segment>-<model index>-<group index>.
import logging
import os
import shlex

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, resolve_sglang_config
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.rollout.session.config import compute_session_server_config
from miles.router.config import compute_miles_router_config
from miles.utils.function_registry import load_function
from miles.utils.http_utils import resolve_ip
from miles.utils.workers.argv_utils import config_to_argv, python_argv_prefix
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.reporter import RegistrationReporter
from miles.utils.workers.types import DeployComponent, PlatformAccess
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.static import StaticWorkerProvider, parse_host_and_port
from miles.utils.workers.worker_spec import (
    CommandWorkerSpec,
    LaunchCommandContext,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
)

logger = logging.getLogger(__name__)

POOL_CATEGORY_INFERENCE_ENGINE = "inference_engine"

ENGINE_POOL_ID_PREFIX = "inference-engine"
INFERENCE_CONTROLLER_ADDR_FLAG = "--inference-controller-addr"
INFERENCE_CONTROLLER_POOL_ID = "inference-controller"
SESSION_SERVER_POOL_ID = "session-server"
INFERENCE_CONTROLLER_WORKER_CLASS = "miles.ray.rollout.inference_controller.InferenceController"
INFERENCE_REGISTRATION_REPORTER_POOL_ID = "inference-registration-reporter"
INFERENCE_REGISTRATION_REPORTER_WORKER_CLASS = "miles.utils.workers.registration.reporter.RegistrationReporterWorker"


def spec_inference_controller(args) -> ServeWorkerSpec:
    # [FUNCTION DOC] Spec of the single inference controller worker: one cell, one worker, no GPUs,
    # one CPU, optionally pinned to the head node by args, read-only platform access, no ports and
    # no extra environment. Constructor arguments (computed lazily from the launch context's backend
    # capability): args, the engine provider (own engines or a registration hub) and the router
    # providers.
    return ServeWorkerSpec(
        name=INFERENCE_CONTROLLER_POOL_ID,
        platform_access=PlatformAccess.READ,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
        worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        ctor_kwargs=lambda ctx: dict(
            args=args,
            engine_provider=_compute_controller_engine_provider(args, capability=ctx.capability),
            router_providers=compute_router_providers(args, capability=ctx.capability),
        ),
    )


def specs_inference_registration_reporter(args) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] Only for an inference-only deploy component, return one spec (otherwise an empty
    # list) for a worker that announces this deployment's engine cells to a remote inference
    # controller: same shape as the controller spec (one cell, one CPU, read access, optional head
    # pin), belongs to the inference component, constructed with args and a registration reporter.
    if DeployComponent(args.deploy_component) is not DeployComponent.INFERENCE:
        return []

    return [
        ServeWorkerSpec(
            name=INFERENCE_REGISTRATION_REPORTER_POOL_ID,
            deploy_component=DeployComponent.INFERENCE,
            platform_access=PlatformAccess.READ,
            port_infos=[],
            env_var=lambda _ctx: {},
            scheduling=SchedulingSpec(
                num_cells=1,
                num_workers_per_cell=1,
                num_gpus_per_worker=0,
                num_cpus_per_worker=1,
                pin_to_head=args.pin_rollout_manager_to_head,
            ),
            worker_class=INFERENCE_REGISTRATION_REPORTER_WORKER_CLASS,
            ctor_kwargs=lambda ctx: dict(
                args=args, reporter=_create_inference_registration_reporter(args, capability=ctx.capability)
            ),
        )
    ]


def _compute_controller_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    # [FUNCTION DOC] HELPER: If this deployment runs its own inference engines, the configured engine
    # provider; otherwise a registration hub for the run (engines of another deployment announce
    # themselves to it).
    if DeployComponent(args.deploy_component).deploys_own_inference_engines():
        return compute_engine_provider(args, capability=capability)
    return RegistrationHub(run_uuid=args.run_uuid)


def _create_inference_registration_reporter(args, *, capability: BackendCapability) -> RegistrationReporter:
    # [FUNCTION DOC] HELPER: Build a reporter for the run (run uuid, reporter id equal to the deploy
    # instance id) whose hub endpoint is the inference controller worker handle (looked up via the
    # controller provider, which may be a static address) and whose worker provider is the engine
    # provider.
    controller_provider = compute_inference_controller_provider(args, capability=capability)
    return RegistrationReporter(
        run_uuid=args.run_uuid,
        reporter_id=args.deploy_instance_id,
        hub_endpoint=controller_provider.get_handle(inference_controller_worker_name()),
        worker_provider=compute_engine_provider(args, capability=capability),
    )


def compute_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    # [FUNCTION DOC] Load the factory named by the custom inference engine provider path in args
    # (extension point; the default is the backend factory below) and call it with args and the
    # capability. Returns the provider.
    return load_function(args.custom_inference_engine_provider_path)(args, capability=capability)


def backend_inference_engine_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    # [FUNCTION DOC] Default engine provider factory: a dynamic worker provider from the backend
    # capability covering all engine pool ids computed from args.
    return capability.dynamic_worker_provider(pool_ids=compute_engine_pool_ids(args))


def compute_router_providers(args, *, capability: BackendCapability) -> list[BaseWorkerProvider]:
    # [FUNCTION DOC] One static worker provider per model in the resolved sglang config, each for
    # that model's router pool id, in model order.
    config = resolve_sglang_config(args)
    return [
        capability.static_worker_provider(pool_id=compute_router_pool_id(model_idx))
        for model_idx in range(len(config.models))
    ]


def create_inference_controller_handle(*, capability: BackendCapability) -> BaseWorkerHandle:
    # [FUNCTION DOC] Handle (RPC proxy) to the inference controller worker obtained from a static
    # provider for the controller pool. Does not contact the worker.
    worker_name = inference_controller_worker_name()
    provider = capability.static_worker_provider(pool_id=INFERENCE_CONTROLLER_POOL_ID)
    return provider.get_handle(worker_name)


def compute_inference_controller_provider(args, *, capability: BackendCapability) -> BaseWorkerProvider:
    # [FUNCTION DOC] If an explicit inference controller address is configured, a static provider
    # over that single RPC address (parsed from host:port) for the controller worker class; else the
    # backend capability's static provider for the controller pool.
    if (entry := args.inference_controller_addr) is not None:
        return StaticWorkerProvider.of_rpc_addrs(
            pool_id=INFERENCE_CONTROLLER_POOL_ID,
            addrs=[parse_host_and_port(entry)],
            worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        )
    return capability.static_worker_provider(pool_id=INFERENCE_CONTROLLER_POOL_ID)


def session_server_worker_name(cell_index: int) -> str:
    # [FUNCTION DOC] Worker name of the session server at the given cell index, via the shared naming
    # helper with the session server pool id.
    return compute_worker_name(pool_id=SESSION_SERVER_POOL_ID, cell_index=cell_index)


def inference_controller_worker_name() -> str:
    # [FUNCTION DOC] Worker name of the (single) inference controller worker.
    return compute_worker_name(pool_id=INFERENCE_CONTROLLER_POOL_ID)


def specs_router(args) -> list[CommandWorkerSpec]:
    # [FUNCTION DOC] One router command spec per model in the resolved sglang config, in model order.
    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    return [
        _compute_spec_router(args, model_idx=model_idx, model_cfg=model_cfg)
        for model_idx, model_cfg in enumerate(config.models)
    ]


def compute_router_pool_id(model_idx: int) -> str:
    # [FUNCTION DOC] Pool id of a model's router: "inference-router-" followed by the model index.
    return f"inference-router-{model_idx}"


def compute_router_worker_name(model_idx: int) -> str:
    # [FUNCTION DOC] Worker name of a model's router (a single-cell pool).
    return compute_worker_name(pool_id=compute_router_pool_id(model_idx))


def _compute_spec_router(args, model_idx: int, model_cfg: ModelConfig) -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Command spec for one model's router: pool id from the model index; ports:
    # a primary port (see port helper) and a prometheus port (static 9000, dynamic allowed); no extra
    # environment; single worker with zero GPUs, optionally pinned to the head node (TODO in code
    # says the flag should be refactored); the launch command is built lazily from the launch context.
    interpreter_prefix = python_argv_prefix()

    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        # [FUNCTION DOC] HELPER: Build the shell-quoted router launch command. Prefill/decode
        # disaggregation applies if the model config has it or the external-router PD flag is set.
        # With the Miles router: asserts no PD disaggregation (assertion error), builds the Miles router
        # config from the primary host and port and the model's engine count, and runs the Miles router
        # module with the config converted to arguments. Otherwise: builds SGLang router arguments
        # (host resolved to an IP, primary port, prometheus port, PD flag), logs them, and runs the
        # sglang router launch module with them. Uses the same python interpreter prefix as this
        # process.
        primary = ctx.self_addrs["primary"]

        has_pd_disaggregation = model_cfg.has_pd_disaggregation or args.rollout_external_router_pd

        if args.use_miles_router:
            assert not has_pd_disaggregation, "miles router does not support PD disaggregation."
            router_config = compute_miles_router_config(
                args, host=primary.host, port=primary.port, num_engines=model_cfg.num_server_cells
            )
            launch_argv = [*interpreter_prefix, "-m", "miles.router.router", *config_to_argv(router_config)]
        else:
            router_args = compute_sglang_router_args(
                args,
                host=resolve_ip(primary.host),
                port=primary.port,
                prometheus_port=ctx.self_addrs["prometheus"].port,
                has_pd_disaggregation=has_pd_disaggregation,
            )
            logger.info(f"Launch router with args: {router_args}")
            launch_argv = [
                *interpreter_prefix,
                "-m",
                "sglang_router.launch_router",
                *router_args_to_argv(router_args),
            ]

        return shlex.join(launch_argv)

    return CommandWorkerSpec(
        name=compute_router_pool_id(model_idx),
        port_infos=[
            _compute_router_primary_port_info(args, model_idx=model_idx),
            PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
        ],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec.single(
            num_gpus_per_worker=0,
            # TODO: refactor the flag
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
        launch_command=_compute_launch_command,
    )


def _compute_router_primary_port_info(args, model_idx: int) -> PortInfo:
    # [FUNCTION DOC] HELPER: Primary port of a model's router: if no router port is configured, static
    # 8000 with dynamic fallback allowed; otherwise the configured port plus the model index, static
    # with no dynamic fallback.
    if args.sglang_router_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.sglang_router_port + model_idx)


def spec_session_server(args) -> CommandWorkerSpec:
    # [FUNCTION DOC] Command spec of the session server pool. Cell count: the configured session
    # server worker count if the session server is enabled and the sglang config has models, else
    # zero cells. One worker per cell, no GPUs or CPUs reserved, one primary port (see port helper).
    # Pinned to the head node if requested by args or if an external host is configured (a single
    # external host only reaches instances on the head).
    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    interpreter_prefix = python_argv_prefix()

    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        # [FUNCTION DOC] HELPER: Build the shell-quoted session server launch command: takes the single
        # router worker address of model 0's router pool (unpacking asserts exactly one) as the backend
        # URL; builds the session server config from args with host equal to the configured server ip
        # or else this worker's primary host, this worker's primary port, and an instance id made from
        # the run uuid and the cell index; runs the session server module with the config as arguments.
        # Note the local name for the config shadows the outer sglang config variable.
        (router_addrs,) = ctx.pool_addrs[compute_router_pool_id(0)]
        config = compute_session_server_config(
            args,
            host=args.session_server_ip or ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            # TODO: make the indexing it k8s native compatible
            instance_id=compute_session_server_instance_id(args, ctx.cell_index),
            backend_url=router_addrs["primary"].addr,
        )
        launch_argv = [*interpreter_prefix, "-m", "miles.rollout.session.server", *config_to_argv(config)]
        return shlex.join(launch_argv)

    return CommandWorkerSpec(
        name=SESSION_SERVER_POOL_ID,
        port_infos=[
            _compute_session_server_primary_port_info(args),
        ],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=(args.session_server_workers if args.use_session_server and config.models else 0),
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=0,
            # one external host for every instance only reaches them while they all sit on the head
            pin_to_head=args.pin_rollout_manager_to_head or bool(args.session_server_external_host),
        ),
        launch_command=_compute_launch_command,
    )


def _compute_session_server_primary_port_info(args) -> PortInfo:
    # [FUNCTION DOC] HELPER: Primary port of session servers: without a configured port, static 8000
    # with dynamic fallback; with one, that static port offset by the cell index so cells on a host
    # do not collide (no dynamic fallback).
    if args.session_server_port is None:
        return PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    return PortInfo(name="primary", static_port=args.session_server_port, offset_by_cell=True)


def compute_session_server_instance_id(args, instance_index: int) -> str:
    # [FUNCTION DOC] Instance id text "<run uuid>-<instance index>".
    return f"{args.run_uuid}-{instance_index}"


def compute_engine_pool_id(args, *, model_idx: int, group_index: int) -> str:
    # [FUNCTION DOC] Pool id "inference-engine-<segment>-<model index>-<group index>", where the
    # segment is the deploy instance id if set, else the deploy component's value.
    segment = args.deploy_instance_id or DeployComponent(args.deploy_component).value
    return f"{ENGINE_POOL_ID_PREFIX}-{segment}-{model_idx}-{group_index}"


def specs_inference_engine(args) -> list[CommandWorkerSpec]:
    # [FUNCTION DOC] Engine pool specs: empty when rollout engines are external. Otherwise one spec for
    # every server group of every model (in model then group order) whose worker type is not
    # placeholder; the group index counts placeholders too.
    if args.rollout_external:
        return []

    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly

    return [
        _compute_spec_inference_engine(
            args,
            model_idx=model_idx,
            group_index=group_index,
            model_cfg=model_cfg,
            server_group_config=server_group_config,
        )
        for model_idx, model_cfg in enumerate(config.models)
        for group_index, server_group_config in enumerate(model_cfg.server_groups)
        if server_group_config.worker_type != "placeholder"
    ]


def compute_engine_pool_ids(args) -> list[str]:
    # [FUNCTION DOC] Names of all engine pool specs, in spec order.
    return [spec.name for spec in specs_inference_engine(args)]


def _compute_spec_inference_engine(
    args,
    model_idx: int,
    group_index: int,
    model_cfg: ModelConfig,
    server_group_config: ServerGroupConfig,
) -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Command spec for one server group of one model. Workers per cell: engine
    # GPUs divided by GPUs per node, at least 1 (one worker per node an engine spans). Asserts the
    # engine GPU count either fits in one node or is a whole multiple of nodes, and that the group's
    # total worker count is a whole number of engines. Scheduling: cells = group GPUs / engine GPUs;
    # 0.2 GPU per worker (TODO: placeholder for real counts in k8s mode); GPU slots per worker =
    # min(engine GPUs, GPUs per node); placement group "rollout" starting at the group's GPU offset.
    # Ports: primary (static 8000), dist_init (master, 9000, 30 + sglang data-parallel-size
    # consecutive), nccl (10000), disaggregation bootstrap (11000, prefill groups only), engine info
    # bootstrap (12000), and a launch gate (master, 13000); all with dynamic fallback. Environment:
    # engine env vars (see helper). Metadata (per cell): model id, worker type, engine GPUs, GPU
    # offset (group offset plus cell index times GPU slots per cell), the API key (group override
    # else args), needs-offload and update-weights flags.
    num_workers_per_cell = max(1, server_group_config.num_gpus_per_engine // args.num_gpus_per_node)
    interpreter_prefix = python_argv_prefix()

    def _compute_launch_command(ctx: LaunchCommandContext) -> str:
        # [FUNCTION DOC] HELPER: Build the engine launch command for one worker (node) of a cell. The
        # random seed is the base seed plus the group's engine offset plus cell index times workers per
        # cell plus the worker's index in the cell (TODO in code: only node 0's seed matters to
        # SGLang). Passes node rank (the worker index in the cell), worker type, the first local GPU id
        # as base GPU, group overrides, engine GPU count, the distributed init address, the nccl port,
        # primary host and port, the optional bootstrap port (prefill only), the engine info bootstrap
        # port and the launch gate port, via the shared engine command builder.
        dist_init = ctx.self_addrs["dist_init"]
        # TODO: only node 0's seed is used by sglang; node != 0 should get node 0's number
        random_seed = (
            args.seed
            + server_group_config.engine_offset
            + ctx.cell_index * num_workers_per_cell
            + ctx.worker_in_cell_index
        )
        return compute_engine_launch_cmd(
            args=args,
            interpreter_prefix=interpreter_prefix,
            # TODO: make the indexing it k8s native compatible
            node_rank=ctx.worker_in_cell_index,
            worker_type=server_group_config.worker_type,
            base_gpu_id=ctx.local_gpu_ids[0],
            sglang_overrides=server_group_config.overrides,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            dist_init_addr=f"{dist_init.host}:{dist_init.port}",
            nccl_port=ctx.self_addrs["nccl"].port,
            host=ctx.self_addrs["primary"].host,
            port=ctx.self_addrs["primary"].port,
            disaggregation_bootstrap_port=d.port if (d := ctx.self_addrs.get("disaggregation_bootstrap")) else None,
            engine_info_bootstrap_port=ctx.self_addrs["engine_info_bootstrap"].port,
            gated_launch_port=ctx.self_addrs[GATE_PORT_NAME].port,
            random_seed=random_seed,
        )

    num_gpus_per_engine = server_group_config.num_gpus_per_engine
    assert num_gpus_per_engine <= args.num_gpus_per_node or num_gpus_per_engine % args.num_gpus_per_node == 0, (
        f"group '{server_group_config.worker_type}' wants {num_gpus_per_engine=} which neither fits in one node of "
        f"{args.num_gpus_per_node} gpus nor tiles whole nodes, so its ranks would never all be launched"
    )

    envs = compute_inference_engine_env_vars(args)
    scheduling = SchedulingSpec(
        num_cells=server_group_config.num_gpus // server_group_config.num_gpus_per_engine,
        num_workers_per_cell=num_workers_per_cell,
        # TODO: may need real num for k8s native mode
        num_gpus_per_worker=0.2,
        num_gpu_slots_per_worker=min(server_group_config.num_gpus_per_engine, args.num_gpus_per_node),
        num_gpus_per_node=args.num_gpus_per_node,
        pg_name="rollout",
        pg_slot_offset=server_group_config.gpu_offset,
    )

    num_workers_total = server_group_config.num_gpus // scheduling.num_gpu_slots_per_worker
    assert num_workers_total % scheduling.num_workers_per_cell == 0, (
        f"group '{server_group_config.worker_type}' has {num_workers_total=} which is not a whole number of "
        f"{scheduling.num_workers_per_cell}-worker engines; the trailing engine would have no node to run its "
        f"remaining ranks"
    )

    return CommandWorkerSpec(
        name=compute_engine_pool_id(args, model_idx=model_idx, group_index=group_index),
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        deploy_component=DeployComponent.INFERENCE,
        port_infos=[
            PortInfo(name="primary", static_port=8000, allow_dynamic=True),
            PortInfo(
                name="dist_init",
                static_port=9000,
                mode="master",
                allow_dynamic=True,
                num_consecutive=30 + args.sglang_dp_size,
            ),
            PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
            *(
                [PortInfo(name="disaggregation_bootstrap", static_port=11000, allow_dynamic=True)]
                if server_group_config.worker_type == "prefill"
                else []
            ),
            PortInfo(name="engine_info_bootstrap", static_port=12000, allow_dynamic=True),
            PortInfo(name=GATE_PORT_NAME, static_port=13000, mode="master", allow_dynamic=True),
        ],
        env_var=lambda _ctx: envs,
        scheduling=scheduling,
        launch_command=_compute_launch_command,
        # TODO: reduce complexity around passing around configs later during arguments refactor
        meta=lambda ctx: dict(
            model_id=model_cfg.name,
            worker_type=server_group_config.worker_type,
            num_gpus_per_engine=server_group_config.num_gpus_per_engine,
            gpu_offset=server_group_config.gpu_offset
            + ctx.cell_index * scheduling.num_workers_per_cell * scheduling.num_gpu_slots_per_worker,
            sglang_api_key=server_group_config.overrides.get("api_key", args.sglang_api_key),
            needs_offload=server_group_config.needs_offload,
            update_weights=model_cfg.update_weights,
        ),
    )


def compute_inference_engine_env_vars(args) -> dict[str, str]:
    # [FUNCTION DOC] Environment for engine workers. Starts with the "do not set visible devices"
    # variables all set to "1". Adds a fixed set of SGLang/NVSHMEM tuning variables, each taking the
    # value already present in the current process environment if any, else a default: NVSHMEM NCCL
    # disabled, deep GEMM precompile off, per-process DG cache dir on, TP memory imbalance check off,
    # memory saver for CUDA graph on, custom all reduce v2 off ("0") when colocated with more than
    # one GPU per engine else on ("1"), batch-invariant matmul fallback on, generation health
    # endpoint off, strict idle memory check off, expose own env vars on. If the dumper is enabled for
    # training or inference, merges the dumper's SGLang environment (imported lazily). Finally forces
    # the original-logprob return variable to "0" so engine log probs match the trainer's
    # temperature-scaled ones, overriding anything inherited or from the dumper. Returns the mapping.
    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
        key: os.environ.get(key, default_val)
        for key, default_val in {
            # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
            "NVSHMEM_DISABLE_NCCL": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": (
                "0" if args.colocate and args.rollout_num_gpus_per_engine > 1 else "1"
            ),
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            "SGLANG_EXPOSE_OWN_ENV_VARS": "1",
        }.items()
    }
    if args.dumper_enable or args.dumper_inference:
        from miles.utils import dumper_utils

        env_vars.update(dumper_utils.get_sglang_env(args))
    # Match the trainer's temperature-scaled logprobs on every worker,
    # overriding inherited environment and dumper settings.
    env_vars["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "0"
    return env_vars
