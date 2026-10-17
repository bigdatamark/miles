# [MODULE DOC] Launcher with two subcommands for running a Tinker-compatible gateway backed by
# Qwen3-30B-A3B with multiple LoRA adapters on one node: prepare downloads the model, serve
# starts the long-running gateway job (the serve script of the repo rather than a training
# script) on disjoint GPU sets for training and for sampling. The gateway idles until clients
# connect; each client gets one of a fixed pool of adapter slots. Options come from a dataclass
# of the shared train config via the command line wrapper. There is also an empty callback so the
# command line app always has subcommands.
"""Prepare Qwen3-30B-A3B and serve the Tinker gateway on separate training and sampling GPUs."""

from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # [CLASS DOC] Options extending the shared train config. Run id (generated per instance),
    # optional HF checkpoint path and checkpoint save dir (filled in after construction), Megatron
    # model type and path, model dir; GPUs per node 8 split into 4 actor and 4 rollout GPUs;
    # tensor parallel 2 and expert parallel 4 for training; LoRA settings (maximum rank 32,
    # alpha 64, number of adapter slots 4, target module list as a comma separated string);
    # gateway port 10613; GPUs per rollout engine 2; sglang static memory fraction 0.7; extra
    # raw arguments.
    run_id: str = field(default_factory=U.create_run_id)

    hf_checkpoint: str | None = None
    model_type: str = "qwen3-30B-A3B"
    model_dir: str = "/root/models"
    save_dir: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    actor_num_gpus: int = 4
    rollout_num_gpus: int = 4
    tp: int = 2
    ep: int = 4

    # LoRA slot pool; per-client rank comes from the SDK, capped by lora_rank.
    lora_rank: int = 32
    lora_alpha: int = 64
    n_adapters: int = 4
    target_modules: str = "attn,mlp,unembed"

    tinker_port: int = 10613
    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.7

    extra_args: str = ""

    def __post_init__(self):
        # [FUNCTION DOC] Fill defaults after construction: the save dir becomes a checkpoints
        # folder under the output dir if unset, and the HF checkpoint becomes the Qwen3-30B-A3B
        # folder under the model dir if unset.
        if self.save_dir is None:
            self.save_dir = f"{self.output_dir}/checkpoints"
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/Qwen3-30B-A3B"


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Subcommand: create the model dir and download the Qwen3-30B-A3B HF
    # checkpoint into a folder of that name under it. No return.
    # KNOWN ISSUE: the download target and repository are fixed, ignoring the HF checkpoint and
    # model type options, so a non-default checkpoint option is never downloaded.
    """Download the Qwen3-30B-A3B checkpoint. Run once per node before serving."""
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir}")
    backend.exec_command_cpu(f"hf download Qwen/Qwen3-30B-A3B --local-dir {args.model_dir}/Qwen3-30B-A3B")


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    # [FUNCTION DOC] Subcommand: print a summary line, build the argument string and start the
    # gateway job. Arguments: checkpoint (HF path, bridge mode for Megatron-to-HF conversion);
    # LoRA (rank, alpha, dropout 0, no fused gradient accumulation, number of adapter slots,
    # quoted target modules); gateway (port, checkpoint root of the save dir plus run id);
    # optimizer Adam with a placeholder learning rate (real values come with each optimizer
    # step request); parallelism (tensor and expert sizes, sequence parallel, dynamic batch of
    # 8192 tokens per GPU); sglang (GPUs per engine, memory fraction, triton LoRA backend);
    # topology (one actor node with the actor GPUs, rollout GPUs); no dropout, fp32 gradient
    # all-reduce and softmax, flash attention; extra args. Submits the repo's serve script (not a
    # train script) with a job lifetime tied to the launcher. No return.
    """Serve the Tinker gateway (idles until clients connect)."""
    print(
        f"[run] tinker gateway: {args.actor_num_gpus} train + {args.rollout_num_gpus} rollout GPUs, "
        f"{args.n_adapters} adapter slots, port {args.tinker_port}"
    )

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
        "--no-gradient-accumulation-fusion "
        f"--multi-lora-n-adapters {args.n_adapters} "
        f'--target-modules "{args.target_modules}"'
    )

    tinker_args = f"--tinker-server-port {args.tinker_port} " f"--tinker-checkpoint-root {args.save_dir}/{args.run_id}"

    # initial config only; AdamParams come per optim_step request
    optimizer_args = "--optimizer adam --lr 1e-4 "

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {args.ep} --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-lora-backend triton "
    )

    topology_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} {lora_args} {tinker_args} {optimizer_args} "
        f"{perf_args} {sglang_args} {topology_args} {misc_args} {args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.model_type,
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
    )


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: Intentionally empty command group callback, present only so the app
    # keeps subcommand names (prepare and serve) instead of collapsing a single command.
    pass


if __name__ == "__main__":
    app()
