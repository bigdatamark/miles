# [MODULE DOC]
# Command line launcher (typer) for GRPO training of GLM-4.7-Flash on Harbor agentic tasks where
# the Harbor trial runs inside each rollout worker against a cloud sandbox backend. Flow of the
# entry point: kill stale training processes, optionally convert the Hugging Face checkpoint to the
# Megatron distributed format, then assemble the training arguments in groups (checkpoint, rollout
# with 4 prompts times 8 samples, performance with tensor parallel 4 and expert parallel 8 and CPU
# optimiser offload, GRPO with a small KL loss, Adam, inference engine with one GPU per engine and
# GLM tool and reasoning parsers, the shared agentic wiring, colocated miscellaneous options,
# optional debug and trace and W&B options) and submit the job with a module search path that
# includes Megatron, this example, the reused agent-server example and the repository, plus
# the Harbor environment variables forwarded to workers. Defaults of the script argument record
# are read from environment variables at import time.
"""Launcher: GLM-4.7-Flash GRPO on Harbor tasks, with Harbor run in-process on a
cloud sandbox backend (no agent server).

Usage:
    HARBOR_ENV_TYPE=e2b python run.py --harbor-tasks-dir /path/to/harbor_tasks ...

The trainer side is examples/swe-agent-harbor-docker/run.py with the agent
server swapped for harbor_agent_function.run; the reward hook and metrics come
from that example (generate.py), which this launcher puts on PYTHONPATH.
"""

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Literal

import typer
from launch_common import agentic_pythonpath_dirs, agentic_train_args, harbor_env_vars

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line argument record (extends the shared execute-train configuration). Groups: mode
    # (normal, or debug rollout only which skips training), run id (generated once at import time),
    # model type and node/GPU counts, Megatron path; paths and model names; training sizes (max
    # sequence length 65536, 3000 rollouts, 4 prompts per rollout, 8 samples per prompt, global batch
    # 32, save interval 100, optional trace dump directory); Harbor settings (backend type, backend
    # keyword JSON, tasks and trials directories, agent model name, agent timeout default 5400
    # seconds) and provider credential file paths (all defaulting from same-named environment
    # variables evaluated at import time); the externally visible session server host and the host
    # IP; and Weights & Biases settings (key from either of two environment variables).
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = command_utils.create_run_id()
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "GLM-4.7-Flash"
    hf_checkpoint: str = "zai-org/GLM-4.7-Flash"
    ref_load: str = "/root/GLM-4.7-Flash_torch_dist"
    save_dir: str = "/root/GLM-4.7-Flash_harbor/"
    prompt_data: str = "/root/tb2_train.jsonl"

    # Training settings
    max_seq_len: int = 65536
    num_rollout: int = 3000
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 100
    save_traces_dir: str = ""

    # Harbor settings (see harbor_agent_function.py for what each does)
    harbor_env_type: str = os.environ.get("HARBOR_ENV_TYPE", "")
    harbor_env_kwargs: str = os.environ.get("HARBOR_ENV_KWARGS", "")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    harbor_trials_dir: str = os.environ.get("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    agent_timeout: int = int(os.environ.get("AGENT_TIMEOUT", "5400"))
    # provider key files; the launcher forwards the PATH, workers read the file
    daytona_api_key_file: str = os.environ.get("DAYTONA_API_KEY_FILE", "")
    e2b_api_key_file: str = os.environ.get("E2B_API_KEY_FILE", "")
    modal_config_file: str = os.environ.get("MODAL_CONFIG_PATH", "")

    # The host agents outside the cluster reach every session server on; passed as
    # --session-server-external-host, which also keeps the session servers on the head node.
    session_server_external_host: str = ""
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "my-wandb-project")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm47-flash-harbor-inprocess"


def cleanup():
    # [FUNCTION DOC] HELPER: Frees GPU resources from earlier runs. For each of the patterns
    # sglang, train.py and MegatronTrain, lists matching processes with the pattern-grep tool,
    # excludes this process and its parent, and sends the default terminate signal to the rest (errors
    # ignored); then sleeps five seconds. Side effect: kills any matching process on the machine,
    # including unrelated ones.
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run(f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true", shell=True)
    time.sleep(5)


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Converts the Hugging Face checkpoint named in the arguments into the Megatron
    # distributed format under the base directory, using the configured GPUs per node and Megatron
    # path. Delegates to the backend's checkpoint converter.
    U = args.create_backend()
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string and submits the job. Groups as described
    # in the module overview. Details: rollout sampling temperature 0.8 with a response length cap of
    # 8192 and the configured maximum sequence length; the actor uses the configured node and GPU
    # counts, and the rollout GPU count equals GPUs per node (colocated); W&B flags are added only if a
    # key is set (with an optional team); the debug flag is added only in debug-rollout-only mode;
    # trace dumping only if a directory is set. Extra environment for the job: module search path
    # (Megatron, agentic directories, repository base), the Harbor variables from the shared
    # preflight (which can raise), and the host IP when set. Submits through the backend with the
    # script arguments as configuration.
    U = args.create_backend()
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )
    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        "--rollout-max-response-len 8192 "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-router-port 31000 "
    )
    agent_args = agentic_train_args(
        tito_model="glm47",
        session_server_workers=32,
        session_server_external_host=args.session_server_external_host,
    )
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_gpus_per_node} "
    )
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    trace_args = f"--dump-details {args.save_traces_dir} " if args.save_traces_dir else ""
    wandb_args = ""
    if args.wandb_key:
        wandb_args = f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.wandb_run_name} --wandb-key {args.wandb_key} "
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    train_args = (
        f"{ckpt_args}{rollout_args}{optimizer_args}{grpo_args}{wandb_args}{trace_args}"
        f"{perf_args}{sglang_args}{agent_args}{misc_args}{debug_args}"
    )

    extra_env_vars = {
        "PYTHONPATH": ":".join([args.megatron_path, *agentic_pythonpath_dirs(), str(command_utils.repo_base_dir)]),
        **harbor_env_vars(args),
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Entry point (command line parsed into the argument record by the decorator).
    # Cleans up stale processes, runs checkpoint preparation unless skipped, then executes training.
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
