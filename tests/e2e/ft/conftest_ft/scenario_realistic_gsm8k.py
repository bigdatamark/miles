# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

# [MODULE DOC] Realistic gsm8k fault-recovery scenario. Trains Qwen2.5-0.5B-Instruct on gsm8k with
# a disaggregated layout (4 training GPUs as 2 cells with context parallel 2, plus 4 single-GPU
# rollout engines) while a background injector crashes trainer cells and rollout engines. The
# run's built-in CI metric checker requires eval accuracy on gsm8k at or above a threshold
# (default 0.55, which must stay identical to the no-fault baseline test), and healing is then
# asserted with the shared random-crash witnesses. Defaults: seed 42, 250 rollouts, mean 600
# seconds between trainer injections and 1200 seconds between rollout injections. The module
# also exposes reusable pieces (prepare, train-arg builder, run orchestration) for other
# scenarios, which inject their own fault forms, extra args and virtual cells.

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    MetricThresholdOption,
    NumRolloutOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.e2e.ft.conftest_ft.execution import (
    DATA_DIR,
    MODEL_DIR,
    get_api_server_args,
    get_fully_async_args,
    get_train_script,
)
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import (
    API_SERVER_PORT,
    FaultInjectorHandle,
    spawn_fault_injector,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import (
    CellFaultForms,
    compute_mean_interval_seconds_of_cell_type,
    create_cell_fault_forms,
)
from tests.e2e.ft.conftest_ft.scenario_random_crash import assert_healing
from tests.fast.cluster_backends import create_backend_for_run

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

app: typer.Typer = typer.Typer()

TEST_NAME: str = "realistic_gsm8k"

FT_COMPONENTS: tuple[str, ...] = ("train", "rollout")
DEFAULT_SEED: int = 42
DEFAULT_NUM_ROLLOUT: int = 250
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 600.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 1200.0

MODEL_NAME: str = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE: str = "qwen2.5-0.5B"
# Same disaggregated layout as the kill_train__dp2_cp2__moe_5layer mode: 2 cells x CP2 on
# 4 training GPUs, plus 4 rollout engines x 1 GPU.
TRAIN_GPUS: int = 4
ROLLOUT_GPUS: int = 4
# Must stay identical to the threshold asserted by the no-fault baseline
# tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py: fault recovery must not cost accuracy.
DEFAULT_METRIC_THRESHOLD: float = 0.55


@dataclass(frozen=True)
class Gsm8kRun:
    # [CLASS DOC] Immutable description of a prepared gsm8k run: API server base url, execution
    # config, dump directory, the full train argument string, and a launcher callable that
    # takes a config and starts training (blocking until it ends).
    base_url: str
    config: command_utils.ExecuteTrainConfig
    dump_dir: str
    train_args: str
    launch: Callable[[command_utils.ExecuteTrainConfig], None]

    @property
    def events_dir(self) -> Path:
        # [FUNCTION DOC] The events sub-directory of the dump directory.
        return Path(self.dump_dir) / EVENTS_DIRNAME


@dataclass(frozen=True)
class Gsm8kOutcome:
    # [CLASS DOC] Immutable result of a finished run: the run description and the (already
    # stopped) fault injector handle holding the event log.
    run: Gsm8kRun
    injector: FaultInjectorHandle


CreateCellFaultFormsFn = Callable[[Gsm8kRun], CellFaultForms]


@app.command(name="run")
def run_ci(
    seed: SeedOption = DEFAULT_SEED,
    num_rollout: NumRolloutOption = DEFAULT_NUM_ROLLOUT,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    metric_threshold: MetricThresholdOption = DEFAULT_METRIC_THRESHOLD,
    fully_async: FullyAsyncOption = False,
) -> None:
    # [FUNCTION DOC] Command and CI entry point. Runs the scenario with the default config, a test
    # name suffixed for fully async, train and rollout FT components, per-cell-type mean intervals
    # from the two crash interval options, standard fault forms for the backend, and no extra
    # train args; then asserts healing with the shared random-crash witnesses and prints a pass
    # message. Training fails (inside the run) if the eval accuracy metric is below threshold.
    test_name: str = f"{TEST_NAME}_fully_async" if fully_async else TEST_NAME
    outcome = run_realistic_gsm8k(
        config=command_utils.default_config(),
        test_name=test_name,
        seed=seed,
        num_rollout=num_rollout,
        metric_threshold=metric_threshold,
        fully_async=fully_async,
        mean_interval_seconds_of_cell_type=compute_mean_interval_seconds_of_cell_type(
            FT_COMPONENTS,
            trainer_crash_interval_seconds=trainer_crash_interval_seconds,
            rollout_crash_interval_seconds=rollout_crash_interval_seconds,
        ),
        create_forms=lambda run: create_cell_fault_forms(base_url=run.base_url, config=run.config),
        build_extra_train_args=lambda _dump_dir: "",
    )

    assert_healing(FT_COMPONENTS, injector=outcome.injector, event_dir=outcome.run.events_dir, context=test_name)

    print(f"Random failure gsm8k accuracy test PASSED ({test_name}, seed={seed}, rollouts={num_rollout})")


def run_realistic_gsm8k(
    *,
    config: command_utils.ExecuteTrainConfig,
    test_name: str,
    seed: int,
    num_rollout: int,
    metric_threshold: float,
    fully_async: bool,
    mean_interval_seconds_of_cell_type: dict[str, float],
    create_forms: CreateCellFaultFormsFn,
    build_extra_train_args: Callable[[str], str],
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    enable_fault_tolerance: bool = True,
) -> Gsm8kOutcome:
    # [FUNCTION DOC] Orchestrates one gsm8k run with fault injection and returns the outcome.
    # Parameters: config; test name; seed; rollout count; accuracy threshold; fully-async flag;
    # mean injection seconds per cell type; a factory building fault forms from the run; a
    # function mapping the dump directory to extra train args; optional supplier of virtual
    # cells; flag to enable fault tolerance (default true). Steps: create the backend; print
    # settings; prepare model and data; remove all four proxy environment variables from the
    # process (both cases); create the dump directory fresh (deleting any previous one so stale
    # events are never read); build train args and append the debug event data location and the
    # extra args; build the run (base url from backend host and fixed port; launcher bound to the
    # train args); start the injector; launch training; always stop and join the injector.
    # Exceptions from training propagate after the injector is stopped.
    U = create_backend_for_run(config)
    print(f"Seed: {seed}, Rollouts: {num_rollout}, Mean injection intervals: {mean_interval_seconds_of_cell_type}")
    print(f"Test: {test_name}, train script: {get_train_script(fully_async=fully_async)}")

    prepare_gsm8k(U)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    dump_dir: str = resolve_dump_dir(test_name, run_id=config.run_id)
    # Start from a clean dump dir so the event analyzer never reads a previous run's
    # stale events (run_training does this for the other scenarios; gsm8k bypasses it).
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    train_args = get_gsm8k_train_args(
        config=config,
        seed=seed,
        num_rollout=num_rollout,
        metric_threshold=metric_threshold,
        fully_async=fully_async,
        test_name=test_name,
        enable_fault_tolerance=enable_fault_tolerance,
    )
    train_args += f"--save-debug-event-data {dump_dir}/{EVENTS_DIRNAME} "
    train_args += build_extra_train_args(dump_dir)

    run = Gsm8kRun(
        base_url=f"http://{U.api_server_host(config)}:{API_SERVER_PORT}",
        config=config,
        dump_dir=dump_dir,
        train_args=train_args,
        launch=partial(_launch_gsm8k, train_args=train_args, fully_async=fully_async),
    )
    injector = spawn_fault_injector(
        base_url=run.base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=create_forms(run),
        get_virtual_cells=get_virtual_cells,
    )

    try:
        run.launch(config)
    finally:
        injector.stop_and_join()

    return Gsm8kOutcome(run=run, injector=injector)


def prepare_gsm8k(U: BaseCommandBackend) -> None:
    # [FUNCTION DOC] Prepares inputs through the given backend: creates model and data
    # directories, downloads the Qwen2.5-0.5B-Instruct model, converts it to a Megatron
    # checkpoint using 4 GPUs (Megatron path from an environment variable, default the root-home
    # checkout), and downloads the gsm8k dataset.
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=TRAIN_GPUS,
        hf_checkpoint=f"{MODEL_DIR}/{MODEL_NAME}",
        dir_dst=MODEL_DIR,
        megatron_path=os.environ.get("MILES_SCRIPT_MEGATRON_PATH", "/root/Megatron-LM"),
    )
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=DATA_DIR)


def get_gsm8k_train_args(
    *,
    config: command_utils.ExecuteTrainConfig,
    seed: int,
    num_rollout: int,
    metric_threshold: float,
    fully_async: bool,
    test_name: str,
    enable_fault_tolerance: bool = True,
) -> str:
    # [FUNCTION DOC] Builds the full train argument string. Groups: checkpoint (HF checkpoint and
    # torch-dist reference load); rollout (gsm8k train data, chat template, shuffling, math reward,
    # the rollout count, batch 32 prompts times 8 samples, response length 1024, temperature 1,
    # oversampling batch 64 with a non-zero reward standard deviation dynamic filter, global batch
    # 256, plus fully-async args when requested); Adam optimizer (learning rate 1e-6, constant,
    # weight decay 0.1); GRPO with clip 0.2/0.28 and zero entropy coefficient; wandb args named by
    # test name with a seed-based run name prefix; performance (context parallel 2, dynamic batch
    # size, 9216 tokens per GPU); eval every 20 rollouts on the gsm8k test set, 1 sample, length
    # 1024, top-k 1; SGLang (4 rollout GPUs, 1 per engine, static memory fraction 0.7, metrics on);
    # API server args and, when fault tolerance is enabled, the FT flags for train and rollout
    # components and the mini FT controller; CI checks (KL checker disabled, accuracy key eval/gsm8k
    # with the given threshold); and miscellaneous (zero dropout, fp32 grad accumulation and
    # softmax, flash attention, one actor node with 4 GPUs).
    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}/ " f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist "

    rollout_args = (
        f"--prompt-data {DATA_DIR}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    ) + get_fully_async_args(fully_async=fully_async)

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data gsm8k {DATA_DIR}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        # Parallelism mirrors the kill_train__dp2_cp2__moe_5layer mode (2 cells x CP2), not
        # the no-fault baseline test.
        "--context-parallel-size 2 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = "--advantage-estimator grpo " "--entropy-coef 0.00 " "--eps-clip 0.2 " "--eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus {ROLLOUT_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-metrics "
    )

    fault_tolerance_args = get_api_server_args(config)
    if enable_fault_tolerance:
        fault_tolerance_args += (
            "--use-fault-tolerance " f"--ft-components {' '.join(FT_COMPONENTS)} " "--mini-ft-controller-enable "
        )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        f"--ci-metric-checker-threshold {metric_threshold} "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {TRAIN_GPUS} "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(f'test_{test_name}.py', run_name_prefix=f'seed{seed}')} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{fault_tolerance_args} "
        f"{ci_args} "
        f"{misc_args} "
    )


def _launch_gsm8k(config: command_utils.ExecuteTrainConfig, *, train_args: str, fully_async: bool) -> None:
    # [FUNCTION DOC] HELPER: Starts training through the config's backend with the given
    # arguments, 8 GPUs per node (train plus rollout GPUs), the scenario's Megatron model type, the
    # chosen train script, and env vars that disable torch compile (avoiding slow, memory-heavy
    # recompiles in respawned cells), turn off ray log dedup, and enable SGLang millisecond logs.
    create_backend_for_run(config).execute_train(
        train_args=train_args,
        num_gpus_per_node=TRAIN_GPUS + ROLLOUT_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            # Same as run_training: a cell respawned after a crash cold-recompiles
            # its first forward, which is slow and memory-heavy enough to OOM.
            "TORCHDYNAMO_DISABLE": "1",
            "RAY_DEDUP_LOGS": "0",
            "SGLANG_LOG_MS": "1",
        },
        train_script=get_train_script(fully_async=fully_async),
    )


if __name__ == "__main__":
    app()
