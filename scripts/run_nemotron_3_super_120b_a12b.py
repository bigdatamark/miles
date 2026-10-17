# [MODULE DOC]
# Launcher script (typer command line app with two commands, one per pod role) for a short GRPO
# smoke-test training of NVIDIA Nemotron-3-Super-120B-A12B (hybrid state-space/attention latent
# mixture of experts) on 2 nodes, Megatron bridge mode (the Hugging Face checkpoint doubles as the
# reference load) with colocated SGLang, on DAPO math with no eval. Command "worker": clears stale
# processes, waits for the head's ray port, joins the cluster and blocks. Command "train": sets the
# head address environment variable, starts the ray head, waits until all GPUs are visible, then
# submits the job. Training only: model and data must exist.
"""NVIDIA Nemotron-3-Super-120B-A12B GRPO training script (2 nodes x 8 GPUs).

=====================

nemotron_h is a hybrid Mamba2 + Attention + latent-MoE architecture (88 layers, 512
experts top-22, moe_latent_size=1024). It loads through NVIDIA ``megatron.bridge``
(``--megatron-to-hf-mode bridge``): ``--ref-load`` points at the HF checkpoint itself, so
no offline torch_dist conversion step is needed.

The two pods take different roles and each runs exactly one command:
  * `worker` waits for the head's ray port, joins the cluster and blocks. It never submits.
  * `train` starts the ray head, waits until the cluster reports every GPU of both nodes,
    and only then submits the job.

=====================

Args:
  --head-ip: IP of the head pod. Both roles need it.
  --num-nodes / --num-gpus-per-node: Cluster shape (default: 2 x 8).
  --num-rollout: Rollout steps (default: 10, i.e. the smoke test).
  --model-dir / --data-dir: Checkpoint / dataset directories. On the cluster these are
      /cluster_public/miles_data/models and /cluster_public/miles_data/datasets.

=====================

  on the worker pod:
    python scripts/run_nemotron_3_super_120b_a12b.py worker --head-ip <head_pod_ip>
  on the head pod:
    python scripts/run_nemotron_3_super_120b_a12b.py train --head-ip <head_pod_ip>
"""

import os
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (output dir, backend creation).
    # Fields: run id, head pod IP (default loopback; both roles need the real value), model name and
    # Megatron model type, node count (overridden default 2), GPUs per node (8), number of rollouts
    # (10), extra raw args, data/model/Megatron directories.
    run_id: str = command_utils.create_run_id()
    head_ip: str = "127.0.0.1"
    model_name: str = "NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    megatron_model_type: str = "nemotron-3-super-120b-a12b"
    num_nodes: int = 2
    num_gpus_per_node: int = 8
    num_rollout: int = 10
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"


def _wait_for_head_port(args: ScriptArgs) -> None:
    # [FUNCTION DOC] HELPER: Polls up to 60 times for the head's ray port (6379) to accept connections:
    # each probe runs a shell port check whose exit status is echoed and read from output (because the
    # command runner raises on non-zero exit); returns at the first success; otherwise prints a waiting
    # message and sleeps 5 seconds. Returns normally after the attempts are exhausted.
    # KNOWN ISSUE: no error is raised on timeout; the caller proceeds to join anyway.
    U = args.create_backend()
    for _ in range(60):
        # exec_command_cpu raises on a non-zero exit, so the probe reports through stdout.
        if U.exec_command_cpu(f"nc -z {args.head_ip} 6379 2>/dev/null; echo $?", capture_output=True).strip() == "0":
            return
        print(f"waiting for head {args.head_ip}:6379 ...")
        U.exec_command_cpu("sleep 5")


def _wait_for_ray_gpus(args: ScriptArgs, expected_gpus: int) -> None:
    # [FUNCTION DOC] HELPER: Polls up to 120 times (5 s apart) the ray cluster status text for the
    # expected total GPU count; on success prints a ready message and stops polling. In every case
    # finishes by printing the cluster status, then returns, so the job is submitted even on timeout
    # (the job's own resource error will then explain the shortfall). Used as the pre-submit callback.
    """The head reports only its own 8 GPUs until the worker joins, and a job submitted then gets one node."""
    U = args.create_backend()
    for _ in range(120):
        if f"{expected_gpus}.0 GPU" in U.exec_command_cpu("ray status 2>/dev/null || true", capture_output=True):
            print(f"[ray] cluster ready: {expected_gpus} GPUs")
            break
        U.exec_command_cpu("sleep 5")
    # Submit either way: the job's own resource error names the shortfall, and this leaves it in the log.
    U.exec_command_cpu("ray status")


def _execute_train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Builds the training argument string and submits the job. Checkpoint: the
    # Hugging Face folder as hf checkpoint and reference load, bridge mode, saves to the output
    # directory checkpoints folder every 20 steps. Rollout: DAPO math, deepscaler reward, chat template,
    # shuffled, rollouts from options, 32 prompts x 4 samples, length 1024, temperature 1, global batch
    # 128, balanced. Parallelism: tensor 4 with sequence parallel, pipeline 2, context 1, expert 8,
    # expert-tensor 1; full uniform recompute; dynamic batch 1024 tokens per GPU; log-prob chunk 128.
    # GRPO with kl loss on at zero coefficient, zero entropy, asymmetric clip. Adam 1e-6 with CPU
    # offload group. SGLang: engine size equal to GPUs per node, memory fraction 0.7, rollout routing
    # replay (needed for MoE). Misc: dropouts off, fp32 accumulation and softmax, auto attention
    # backend, colocate, node and GPU counts, rollout GPU total. Submits via the backend with a
    # pre-submit callback waiting for all GPUs to join ray.
    # KNOWN ISSUE: the rollout GPU count flag is passed together with colocate, where it is ignored.
    U = args.create_backend()
    total_gpus = args.num_nodes * args.num_gpus_per_node
    hf_checkpoint = f"{args.model_dir}/{args.model_name}"

    ckpt_args = (
        # --ref-load at an HF directory routes load_checkpoint through megatron.bridge,
        # so the run needs no torch_dist copy of the weights.
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load {hf_checkpoint} "
        f"--save {args.output_dir}/checkpoints "
        "--save-interval 20 "
        "--megatron-to-hf-mode bridge "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 128 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 1024 "
        "--log-probs-chunk-size 128 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        "--sglang-mem-fraction-static 0.7 "
        # Replay the exact rollout routing during the training forward so train logprobs
        # match rollout logprobs (needed for MoE).
        "--use-rollout-routing-replay "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend auto "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {total_gpus} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        before_ray_job_submit=lambda: _wait_for_ray_gpus(args=args, expected_gpus=total_gpus),
        megatron_path=args.megatron_path,
    )


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] Command "train" (head role): sets the master address environment variable of this
    # process to the head IP (the backend reads it for the ray head address and the distributed
    # rendezvous), then runs training.
    """Head role: start the ray head, wait for both nodes' GPUs, submit the job."""
    # execute_train reads MASTER_ADDR for the ray head's node ip and for the torch
    # distributed rendezvous the worker's ranks connect to.
    os.environ["MASTER_ADDR"] = args.head_ip
    _execute_train(args)


@app.command()
@command_utils.dataclass_cli
def worker(args: ScriptArgs):
    # [FUNCTION DOC] Command "worker": force-kills leftover sglang, ray and miles processes from a
    # previous run (ray refuses to join with stale agents), waits for the head port, then starts ray
    # joining the head at port 6379 with the configured GPU count, usage stats disabled, and blocks
    # until the cluster is stopped. Never submits a job.
    """Worker role: join the head's ray cluster and block."""
    # A re-run inherits the previous run's agents, and ray refuses to join with them alive.
    U = args.create_backend()
    U.exec_command_cpu("pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 miles; sleep 3; true; ")
    _wait_for_head_port(args)
    U.exec_command_cpu(
        f"ray start --address={args.head_ip}:6379 "
        f"--num-gpus={args.num_gpus_per_node} "
        "--disable-usage-stats "
        "--block"
    )


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: Empty typer callback so commands stay sub-commands. Does nothing.
    pass


if __name__ == "__main__":
    app()
