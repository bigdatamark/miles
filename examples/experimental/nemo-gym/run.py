# [MODULE DOC] Self-executing launcher that trains Qwen3-4B-Instruct-2507 with GRPO against a
# running NeMo Gym agent server (SWE-bench style tasks). Steps in main: clean up stale processes,
# optionally convert the HF checkpoint to the Megatron distributed format (prepare), then build the
# training argument string in grouped blocks and submit the job through the shared command
# utilities (execute). The agent function, reward hook and session server flags are wired here;
# the NeMo Gym server itself must already be running and reachable at the configured url.
"""NeMo Gym launcher (Qwen3-4B-Instruct-2507): Miles <-> mini_swe_agent_2 orchestration.

Defaults are the exact configuration of the validated smoke run (4x H200,
2026-07-28): 3 GRPO steps at tiny scale against a NeMo Gym server running the
docker sandbox provider. Scale up --num-rollout / batch sizes for real
training.

Usage:
    NEMO_GYM_URL=http://<nemo-gym-host>:12000 python run.py
    python run.py --mode debug_rollout_only
    python run.py --skip-prepare --prompt-data /my/data.jsonl
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command-line configuration (extends the shared train config). Fields: mode
    # (normal, or debug_rollout_only which runs only rollouts); model type, GPUs per node (4) and
    # Megatron path; skip_prepare; base_dir, model_name, HF checkpoint id, ref-load checkpoint
    # path, save dir and prompt data path (all defaulting under /root); training scale (max
    # sequence length 16384, max response length 4096, 3 rollouts, rollout batch 2, 4 samples per
    # prompt, global batch 8, save interval 1000); the NeMo Gym server url (default from the
    # environment variable NEMO_GYM_URL at import time, else http://localhost:12000); and an
    # optional trainer host name that the NeMo Gym host should dial back, forwarded as the session
    # server external host (empty means unset).
    # KNOWN ISSUE: the NeMo Gym url default is evaluated once at import, not at call time.
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    megatron_model_type: str = "qwen3-4B-Instruct-2507"
    num_gpus_per_node: int = 4
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "Qwen3-4B-Instruct-2507"
    hf_checkpoint: str = "Qwen/Qwen3-4B-Instruct-2507"
    ref_load: str = "/root/Qwen3-4B-Instruct-2507_torch_dist"
    save_dir: str = "/root/Qwen3-4B-Instruct-2507_nemogym/"
    prompt_data: str = "/root/swe_verified.jsonl"

    # Training settings (validated smoke scale)
    max_seq_len: int = 16384
    rollout_max_response_len: int = 4096
    num_rollout: int = 3
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    save_interval: int = 1000

    # NeMo Gym settings
    nemo_gym_url: str = os.environ.get("NEMO_GYM_URL", "http://localhost:12000")
    # The trainer host the NeMo Gym host reaches, needed only when that host cannot resolve the trainer's
    # hostname (e.g. it dials back over a tailnet). Passed as --session-server-external-host, which also
    # keeps the session servers on the head node.
    session_server_external_host: str = ""


def cleanup():
    # [FUNCTION DOC] HELPER: Kills leftover processes whose command line matches sglang, train.py
    # or MegatronTrain, using a shell pipeline of process listing, filtering out this process id
    # and its parent id, then kill; failures are ignored. Then sleeps 5 seconds to let GPUs free.
    # Side effects: runs shell subprocesses and terminates other processes on the machine.
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Converts the HF checkpoint into the Megatron distributed format under base_dir
    # through the backend's convert-checkpoint command (skipped by the utility if already done).
    # Side effects: runs conversion commands, uses GPUs.
    """Convert the HF checkpoint to torch_dist format if not already done."""
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
    # [FUNCTION DOC] Builds the training argument string and submits the job. Blocks: checkpoint
    # (HF checkpoint, reference load, save path and interval); rollout (prompt data with input key
    # prompt and metadata key metadata, shuffle, rollout counts, temperature 0.8, response and
    # sequence limits, global batch size, data balancing); GRPO with KL loss (coefficient 0.01, low
    # variance type), zero entropy coefficient and asymmetric clipping 0.2/0.28; Adam optimizer
    # (learning rate 1e-6 constant, weight decay 0.1, betas 0.9/0.98); performance (tensor
    # parallel 2, other parallel sizes 1, full uniform recompute of 1 layer, dynamic batch size
    # with max tokens per GPU equal to max sequence length); SGLang (1 GPU per engine, static
    # memory fraction 0.7); agent (the agentic tool-call generate function, this directory's agent
    # function and reward hook, the no-aborted dynamic sampling filter, session server on 0.0.0.0
    # port 30000 with 32 workers, optional external host, qwen3 TITO model); misc (dropouts off,
    # fp32 grad accumulation and softmax, flash attention backend, colocated actor on one node);
    # plus the rollout-only debug flag in debug mode. The runtime environment gets a python path of
    # Megatron, this directory and the repo base, and the NeMo Gym url. Submits via the backend's
    # execute-train; side effects are starting Ray and the training job.
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
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
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

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_seq_len} "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "

    external_host_arg = (
        f"--session-server-external-host {args.session_server_external_host} "
        if args.session_server_external_host
        else ""
    )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path nemogym_agent_function.run "
        "--custom-rm-path nemogym_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--use-session-server "
        # 0.0.0.0 so the NeMo Gym host can dial in on any interface (e.g. a
        # tailnet address); internal calls resolve it to localhost.
        "--session-server-ip 0.0.0.0 "
        "--session-server-port 30000 "
        "--session-server-workers 32 "
        f"{external_host_arg}"
        "--tito-model qwen3 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
    )

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{grpo_args}"
        f"{optimizer_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{command_utils.repo_base_dir}",
        "NEMO_GYM_URL": args.nemo_gym_url,
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Entry point (command line wired from the config dataclass): runs cleanup,
    # runs prepare unless skip_prepare is set, then runs execute.
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
