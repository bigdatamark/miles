# [MODULE DOC] Launcher for a small Qwen3-0.6B GRPO run used to exercise hot restart: it saves a
# checkpoint every few rollouts and loads from the same directory, so re-running the same run
# id resumes from the latest save. Command line program; prepare stages the model, GSM8K data
# and converted checkpoint; execute submits the job. Reward model is a deterministic random
# type (a test reward, not a real grader).
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command line options extending the shared train config: model name and Megatron
    # type, number of rollouts (20), save interval in rollouts (2), actor GPU count (4), number
    # of rollout engines (2), GPUs per engine (1), data/model/Megatron directories and free-form
    # extra arguments. Run id and output dir come from the base config.
    model_name: str = "Qwen3-0.6B"
    megatron_model_type: str = "qwen3-0.6B"
    num_rollout: int = 20
    save_interval: int = 2
    actor_num_gpus: int = 4
    num_engines: int = 2
    gpus_per_engine: int = 1
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    @property
    def num_gpus_per_node(self) -> int:
        # [FUNCTION DOC] Total GPUs needed on the node: actor GPUs plus engines times GPUs per
        # engine (non-colocated layout). Returns an integer.
        return self.actor_num_gpus + self.num_engines * self.gpus_per_engine

    @property
    def checkpoint_dir(self) -> str:
        # [FUNCTION DOC] Directory used both to save and load training checkpoints: the output
        # dir, then the run id, then a checkpoints folder. Returns a string path.
        return f"{self.output_dir}/{self.run_id}/checkpoints"


def prepare(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Stage inputs: create directories, download the HF model and the GSM8K
    # dataset, and convert the HF checkpoint to a distributed Megatron checkpoint using one
    # process per actor GPU (not per node GPU). No return.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.actor_num_gpus,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Submit the training job with the argument string from the builder, using the
    # derived node GPU count. Uses the default train script (synchronous). No return.
    args.create_backend().execute_train(
        train_args=build_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


def build_train_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] Compose the training argument string and return it. Checkpoint group: HF
    # model, reference load from the converted checkpoint, and save plus load both pointing at
    # the same checkpoint dir with the save interval (this is what enables resume). Rollout
    # group: GSM8K parquet with messages key, chat template, shuffle, deterministic random
    # reward, max response 200, temperature 0.8, 32 prompts x 8 samples per rollout, rollout
    # count and rollout GPU/engine sizing from the options. Optimizer: Adam lr 1e-6 constant,
    # weight decay 0.1, betas 0.9/0.98, fp32 gradient all-reduce. Misc: no dropout, flash
    # attention, context parallel 2, one actor node, dynamic batch with 32768 tokens per GPU,
    # GRPO with clip 0.2, global batch 256. Appends default wandb args, the run id as wandb run
    # id, and the free-form extra args. Parts are joined by single spaces.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--save {args.checkpoint_dir} "
        f"--load {args.checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deterministic_random "
        "--rollout-max-response-len 200 "
        "--rollout-temperature 0.8 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-num-gpus {args.num_engines * args.gpus_per_engine} "
        f"--rollout-num-gpus-per-engine {args.gpus_per_engine} "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--accumulate-allreduce-grads-in-fp32 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--context-parallel-size 2 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus} "
        "--global-batch-size 256 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
        "--advantage-estimator grpo "
        "--eps-clip 0.2 "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{misc_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"--wandb-run-id {args.run_id} "
        f"{args.extra_args} "
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Command line entry: run prepare then execute with the parsed options.
    prepare(args)
    execute(args)


# TODO: unify this launcher when the example scripts are refactored
if __name__ == "__main__":
    typer.run(main)
