# [MODULE DOC] HELPER: Shared test fixtures for the values tests. Defines a sample launch plan
# constant (frozen run id, an orchestrator state file path, release name r, namespace rl, a
# two-word orchestrator command and a two-word worker argv) and factory functions that build
# worker specs: a router, an inference engine, a trainer, a session server and a session client.
# All are pure constructors with no side effects.
from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release="r",
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
)


def router() -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Returns a command worker spec named inference router 0: one primary
    # port fixed at 8000, no env vars, a single zero GPU worker, and a launch command running the
    # router module bound to its own primary address host.
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: f"python -m router --host {ctx.self_addrs['primary'].host}",
    )


def engine(
    num_cells: int = 2,
    gpus_per_engine: int = 32,
    name: str = "inference-engine-0-0",
    gpu_offset: int = 0,
) -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Returns an inference engine command worker spec. Inputs: number of
    # cells (default 2), GPUs per engine (default 32), pool name (default inference engine 0 0) and
    # GPU offset (default 0, used as the placement group slot offset). Workers per cell is the GPU
    # count divided by 8 using integer division, at least 1; GPU slots per worker is the smaller
    # of the GPU count and 8; 0.2 GPU per worker and 8 GPUs per node are fixed. Ports: primary
    # 8000, master mode dist_init 9000, and engine info bootstrap 12000. Env: one var disabling
    # NVSHMEM over NCCL. The launch command starts the sglang server with the worker's index in
    # its cell as node rank, the dist init address and the first assigned GPU id as base GPU id.
    return CommandWorkerSpec(
        name=name,
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[
            PortInfo(name="primary", static_port=8000),
            PortInfo(name="dist_init", static_port=9000, mode="master"),
            PortInfo(name="engine_info_bootstrap", static_port=12000),
        ],
        env_var=lambda ctx: {"NVSHMEM_DISABLE_NCCL": "1"},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=max(1, gpus_per_engine // 8),
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=min(gpus_per_engine, 8),
            num_gpus_per_node=8,
            pg_slot_offset=gpu_offset,
        ),
        launch_command=lambda ctx: (
            f"python -m sglang.launch_server --node-rank {ctx.worker_in_cell_index} "
            f"--dist-init-addr {ctx.self_addrs['dist_init'].host}:{ctx.self_addrs['dist_init'].port} "
            f"--base-gpu-id {ctx.gpu_ids[0]}"
        ),
    )


def trainer(num_cells: int = 2, gpus_per_cell: int = 16) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Returns a trainer serve worker spec named trainer engine actor.
    # Inputs: number of cells (default 2) and GPUs per cell (default 16, used as workers per
    # cell). Fixed: 0.4 GPU per worker, 1 GPU slot per worker, 8 GPUs per node, master mode port
    # 9000, an env var disabling NCCL cumem, the Megatron train actor worker class and empty
    # constructor kwargs.
    return ServeWorkerSpec(
        name="trainer-engine-actor",
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda ctx: {"NCCL_CUMEM_ENABLE": "0"},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=8,
        ),
        worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
        ctor_kwargs=lambda ctx: {},
    )


def session_server(num_cells: int) -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Returns a command worker spec named session server with the given
    # number of cells, one worker per cell, no GPUs and no GPU slots, primary port 8000, no env
    # vars, and a fixed launch command that runs the session server module.
    return CommandWorkerSpec(
        name="session-server",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=1, num_gpus_per_worker=0, num_gpu_slots_per_worker=0
        ),
        launch_command=lambda ctx: "python -m session_server",
    )


def session_client() -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Returns a single worker, zero GPU command spec named rollout
    # executor with primary port 8100 and no env vars. Its launch command runs the executor module
    # with a comma separated list of the primary addresses of every worker in the session server
    # pool, read from the launch context's pool addresses at command build time.
    return CommandWorkerSpec(
        name="rollout-executor",
        port_infos=[PortInfo(name="primary", static_port=8100)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: "python -m executor --session-servers "
        + ",".join(addrs["primary"].addr for addrs in ctx.pool_addrs["session-server"]),
    )
