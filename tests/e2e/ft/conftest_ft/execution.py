# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] Execution helpers for FT end-to-end tests: model/data preparation, assembly of the
# training argument string, environment variables for deterministic runs, and the training
# launcher. At import time it creates a unique temporary run directory holding the Megatron
# source patcher config path, reads the Megatron path from an environment variable (default a
# root-home Megatron-LM checkout), and resolves the model and data directories.

import json
import os
import shutil
import tempfile
from pathlib import Path

from tests.e2e.common_dirs import get_test_data_dir, get_test_model_dir
from tests.e2e.conftest_dumper import MEGATRON_PATCHER_YAMLS
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import API_SERVER_PORT
from tests.e2e.ft.conftest_ft.modes import DEBUG_ROLLOUT_DATA_HF_REPO, FTTestMode
from tests.fast.cluster_backends import create_backend_for_run

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

_RUN_DIR: Path = Path(tempfile.mkdtemp(prefix="ft_test_dumper_"))
_MEGATRON_SOURCE_PATCHER_CONFIG_PATH: Path = _RUN_DIR / "megatron_source_patcher.yaml"
_MEGATRON_PATH: str = os.environ.get("MILES_SCRIPT_MEGATRON_PATH", "/root/Megatron-LM")
MODEL_DIR: str = get_test_model_dir()
DATA_DIR: str = get_test_data_dir()
_DEBUG_ROLLOUT_DATA_DIR: str = f"{DATA_DIR}/{DEBUG_ROLLOUT_DATA_HF_REPO.split('/')[-1]}"


def materialize_cyclic_debug_rollout_data(count: int) -> str:
    # [FUNCTION DOC] Builds a directory of count rollout files by symlinking, cyclically, to the
    # downloaded debug rollout data files (those with all-digit stems, sorted numerically). File i
    # links to the available file at index i modulo the number available. Creates a fresh temp
    # directory inside the data directory with permissions 755 and returns its path. Raises a
    # file-not-found error if no numbered rollout files exist.
    src = Path(_DEBUG_ROLLOUT_DATA_DIR)
    available = sorted(int(p.stem) for p in src.glob("*.pt") if p.stem.isdigit())
    if not available:
        raise FileNotFoundError(f"No debug rollout data files found in {src}")
    dst = Path(tempfile.mkdtemp(prefix="ft_cyclic_rollout_", dir=DATA_DIR))
    dst.chmod(0o755)
    for i in range(count):
        (dst / f"{i}.pt").symlink_to(src / f"{available[i % len(available)]}.pt")
    return str(dst)


def _get_hf_num_layers(model_path: str) -> int:
    # [FUNCTION DOC] HELPER: Reads the config file in the model directory and returns its hidden
    # layer count. Raises if the file or key is missing.
    with open(f"{model_path}/config.json") as f:
        return json.load(f)["num_hidden_layers"]


def prepare(mode: FTTestMode, *, config: command_utils.ExecuteTrainConfig | None = None) -> None:
    # [FUNCTION DOC] Prepares everything a run needs, using the backend for the given (or default)
    # config. Steps: create the model and data directories; download the mode's Hugging Face model
    # into the model directory; convert it to a Megatron checkpoint using the smaller of the
    # train GPUs per node and the model layer count as the conversion GPU count; when the mode has
    # no real rollout, download the debug rollout dataset; download the gsm8k dataset; finally
    # write the thd-variant Megatron source patcher YAML to the run directory file.
    # Side effects: network downloads, subprocess commands, GPU conversion, file writes.
    config = _resolve_config(config)

    U = create_backend_for_run(config)
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download {mode.model_hf_repo} --local-dir {MODEL_DIR}/{mode.model_name}")

    hf_model_path = f"{MODEL_DIR}/{mode.model_name}"
    num_layers = _get_hf_num_layers(hf_model_path)
    convert_gpus = min(mode.train_gpus_per_node, num_layers)

    U.convert_checkpoint(
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_gpus_per_node=convert_gpus,
        megatron_path=_MEGATRON_PATH,
        hf_checkpoint=hf_model_path,
        dir_dst=MODEL_DIR,
    )
    if not mode.has_real_rollout:
        U.hf_download_dataset(DEBUG_ROLLOUT_DATA_HF_REPO, data_dir=DATA_DIR)
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=DATA_DIR)

    megatron_yaml: str = MEGATRON_PATCHER_YAMLS["thd"]
    _MEGATRON_SOURCE_PATCHER_CONFIG_PATH.write_text(megatron_yaml)


def _resolve_config(config: command_utils.ExecuteTrainConfig | None) -> command_utils.ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Returns the given config if truthy, otherwise the default config.
    return config or command_utils.default_config()


def get_common_train_args(
    mode: FTTestMode,
    *,
    dump_dir: str,
    num_steps: int | None = None,
    enable_dumper: bool = True,
    debug_rollout_data_dir: str | None = None,
) -> str:
    # [FUNCTION DOC] Builds the full space-separated training argument string for a mode.
    # Inputs: mode, dump directory, optional step count (defaults to the mode's own count),
    # dumper flag, optional debug rollout data directory (defaults to the downloaded one).
    # Composition: checkpoint args (HF checkpoint and reference load of the torch-dist converted
    # model); Adam optimizer with constant learning rate 1e-6, weight decay 0.1, betas 0.9/0.98,
    # no warmup, fp32 gradient accumulation; rollout args; debug dump args; the mode's parallel
    # args; misc args (zero dropout, flash attention, actor nodes and GPUs from the mode, global
    # batch 256, dynamic batch size with 32768 max tokens per GPU, alltoall MoE dispatcher, GRPO
    # advantage estimator, clip 0.2, number of rollouts); default wandb args for this file.
    # Rollout args: without real rollout, loads debug rollout data files templated by rollout id
    # with train-only debugging, batch 32 prompts times 8 samples. With real rollout, uses chat
    # template messages, label key, shuffling, a deterministic random reward model, max response
    # length 200, temperature 0.8, the same batch shape, the deterministic rollout flags, saves
    # rollout data under the dump directory, and sets rollout GPUs total and per engine, plus
    # colocation when the mode is colocated.
    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{mode.model_name} --ref-load {MODEL_DIR}/{mode.model_name}_torch_dist "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--lr-warmup-fraction 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
    )

    rollout_args: str
    if not mode.has_real_rollout:
        rollout_dir = debug_rollout_data_dir or _DEBUG_ROLLOUT_DATA_DIR
        rollout_args = (
            f"--prompt-data {DATA_DIR}/gsm8k/train.parquet "
            f"--load-debug-rollout-data {rollout_dir}/{{rollout_id}}.pt "
            "--debug-train-only "
            "--rollout-batch-size 32 "
            "--n-samples-per-prompt 8 "
        )
    else:
        rollout_args = (
            f"--prompt-data {DATA_DIR}/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type deterministic_random "
            "--rollout-max-response-len 200 "
            "--rollout-temperature 0.8 "
            "--rollout-batch-size 32 "
            "--n-samples-per-prompt 8 "
            # Required for reproducibility (ref: https://github.com/THUDM/slime/pull/370)
            + DETERMINISTIC_ROLLOUT_ARGS + f"--save-debug-rollout-data {dump_dir}/rollout_data/{{rollout_id}}.pt "
            f"--rollout-num-gpus {mode.total_rollout_gpus} "
            f"--rollout-num-gpus-per-engine {mode.rollout_gpus_per_engine} " + ("--colocate " if mode.colocate else "")
        )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {mode.train_num_nodes} "
        f"--actor-num-gpus-per-node {mode.train_gpus_per_node} "
        f"--global-batch-size 256 "
        "--delay-split-train-data-by-dp "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
        "--moe-token-dispatcher-type alltoall "
        "--advantage-estimator grpo "
        "--eps-clip 0.2 "
        f"--num-rollout {num_steps if num_steps is not None else mode.num_steps} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{optimizer_args} "
        f"{rollout_args} "
        f"{get_debug_dump_args(dump_dir=dump_dir, enable_dumper=enable_dumper)} "
        f"{mode.parallel_args} "
        f"{misc_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
    )

    return train_args


def get_debug_dump_args(*, dump_dir: str, enable_dumper: bool) -> str:
    # [FUNCTION DOC] Returns argument text that always saves debug event data under the dump
    # directory's events folder, and, when the dumper is enabled, also sets the dumper directory
    # (dump dir / dumps), enables forward/backward dumping with model values and grads and the
    # parallel rank in file names, and points the train source patcher at the generated YAML.
    dumper_args: str = ""
    if enable_dumper:
        dumper_args = (
            f"--dumper-dir {dump_dir}/dumps "
            f"--dumper-fwd-bwd enable=1 enable_model_value=1 enable_model_grad=1 include_parallel_rank_in_filename=1 "
            f"--dumper-source-patcher-config-train {_MEGATRON_SOURCE_PATCHER_CONFIG_PATH} "
        )

    return f"--save-debug-event-data {dump_dir}/{EVENTS_DIRNAME} {dumper_args}"


def get_ft_args(mode: FTTestMode) -> str:
    # [FUNCTION DOC] Returns arguments enabling fault tolerance for the mode's FT components, with
    # the API server port set to 0 (let the system pick).
    return f"--use-fault-tolerance --ft-components {' '.join(mode.ft_components)} --api-server-port 0 "


def get_api_server_args(config: command_utils.ExecuteTrainConfig | None = None) -> str:
    # [FUNCTION DOC] Returns API server arguments with the fixed entrypoint port. For the
    # Kubernetes cluster backend it also binds the host to all interfaces (0.0.0.0); otherwise
    # only the port is set. Uses the default config when none is given.
    resolved = config if config is not None else command_utils.default_config()
    if resolved.cluster_backend is not ClusterBackend.KUBERNETES:
        return f"--api-server-port {API_SERVER_PORT} "
    return f"--api-server-port {API_SERVER_PORT} --api-server-host 0.0.0.0 "


DEFAULT_TRAIN_SCRIPT: str = "train.py"
FULLY_ASYNC_TRAIN_SCRIPT: str = "train_async.py"


def get_train_script(*, fully_async: bool) -> str:
    # [FUNCTION DOC] Returns the async train script name when fully asynchronous, else the default
    # train script name.
    return FULLY_ASYNC_TRAIN_SCRIPT if fully_async else DEFAULT_TRAIN_SCRIPT


def get_fully_async_args(*, fully_async: bool) -> str:
    # [FUNCTION DOC] Returns an empty string when not fully async; otherwise arguments enabling
    # fully-async mode with in-place generation pausing.
    if not fully_async:
        return ""
    return "--fully-async --pause-generation-mode in_place "


DETERMINISTIC_ROLLOUT_ARGS: str = (
    "--sglang-enable-deterministic-inference --sglang-attention-backend flashinfer --deterministic-mode "
)

# Required for reproducibility (ref: https://github.com/THUDM/slime/pull/370)
_DETERMINISTIC_ENV_VARS: dict[str, str] = {
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # The default 4096 split overflows FlashInfer's fixed 2 GiB deterministic workspace
    # while capturing the 8192-token prefill graph for the 5-layer Qwen3 MoE model.
    "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE": "8192",
}


def get_train_env_vars_arg(
    mode: FTTestMode, *, deterministic: bool, extra_env_vars: dict[str, str] | None = None
) -> str:
    # [FUNCTION DOC] Builds the train environment-variables argument. Starts from an empty mapping;
    # when deterministic, adds the deterministic variables (ring NCCL algorithm, disallow
    # nondeterministic transformer-engine algorithms, cuBLAS workspace config, FlashInfer
    # prefill split tile size 8192); when the mode has a real non-colocated rollout, enables
    # expandable CUDA allocator segments; then merges extras (which win). Returns an empty string
    # if the mapping is empty, otherwise an argument carrying the mapping as quoted JSON text.
    env_vars: dict[str, str] = {}
    if deterministic:
        env_vars.update(_DETERMINISTIC_ENV_VARS)
    if mode.has_real_rollout and not mode.colocate:
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if extra_env_vars is not None:
        env_vars.update(extra_env_vars)
    if not env_vars:
        return ""
    return f"--train-env-vars '{json.dumps(env_vars)}' "


def run_training(
    train_args: str,
    mode: FTTestMode,
    *,
    dump_dir: str | None = None,
    extra_env_vars: dict[str, str] | None = None,
    config: command_utils.ExecuteTrainConfig | None = None,
    train_script: str = DEFAULT_TRAIN_SCRIPT,
) -> None:
    # [FUNCTION DOC] Launches training through the config's backend. If a dump directory is given
    # and exists, it is deleted first. Environment variables passed to the run are the
    # deterministic set (always applied here, regardless of mode), eager mode (torch compile
    # disabled, to avoid slow recompiles after crash respawns), ray log dedup off, SGLang
    # millisecond logging, then caller extras (which win). Executes the train script with the
    # arguments, node GPU count and Megatron model type from the mode, and the Megatron path.
    U = _resolve_config(config).create_backend()
    if dump_dir is not None and os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    merged_env_vars = {
        **_DETERMINISTIC_ENV_VARS,
        # Run eager (no torch.compile). A cell respawned after a crash cold-recompiles its first
        # forward; under dynamic batch sizes that is a per-shape Inductor compile that is slow
        # (observed 124s..1510s, growing) and memory-heavy enough to OOM-kill the actor. That
        # recompile-on-respawn is a torch.compile + FT infra limitation orthogonal to what these
        # tests assert (FT crash recovery + baseline-vs-target metric equivalence); both runs are
        # eager so the comparison stays valid.
        #
        # TODO: this only sidesteps the respawn recompile cost, it does not fix it. Investigate
        # keeping torch.compile under FT respawn (warm/shared Inductor cache survivor->respawn, or
        # bounded recompile) so the tests can exercise the compiled path again.
        "TORCHDYNAMO_DISABLE": "1",
        "RAY_DEDUP_LOGS": "0",
        "SGLANG_LOG_MS": "1",
        **(extra_env_vars or {}),
    }
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=mode.total_node_gpus,
        megatron_model_type=mode.megatron_model_type,
        extra_env_vars=merged_env_vars,
        megatron_path=_MEGATRON_PATH,
        train_script=train_script,
    )
