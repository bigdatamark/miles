# [MODULE DOC] Launcher for PPO (actor plus critic) training of Qwen3-4B with the Megatron
# backend on a single node, colocating training and rollout on the same GPUs. Command line
# program: prepare downloads the model and the DAPO math dataset and converts the checkpoint;
# execute builds the argument string and submits the synchronous train script. The critic shares
# the actor's GPUs and parallel layout. The comment block below explains the design.
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

# PPO (actor + critic) with the Megatron backend on a single node.
#
# Unlike GRPO, which derives its baseline from a group of samples per prompt, PPO trains a
# separate value model (the critic) and turns rewards into advantages with GAE. The critic is
# colocated on the actor's train GPUs and shares its parallelism, so a PPO run needs no extra
# GPUs over the GRPO equivalent -- it pays in memory instead, which is why --offload-train is
# forced on.
#
# python run_qwen3_4b_ppo.py


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command line options extending the shared train config: run id (generated at
    # import), model name and Megatron type, GPUs per node (4; must equal the product of the
    # tensor, pipeline and context parallel sizes written in the argument string, here 1x2x2),
    # data/model/Megatron directories and free-form extra arguments.
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: str = "qwen3-4B"
    # actor world size, and therefore the critic's too: must equal TP * PP * CP below.
    num_gpus_per_node: int = 4
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Stage inputs: create directories, download the HF model and the DAPO math
    # dataset, and convert the HF checkpoint to a distributed Megatron checkpoint using one
    # process per GPU. No return.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Build the training arguments and submit the job. Checkpoint: HF model,
    # reference load from the converted checkpoint, load and save to the same run folder under the
    # output dir, save every 20 steps; no critic path flags (they default from the actor ones, the
    # critic save dir getting a suffix). Rollout: DAPO math data, deepscaler reward, 300 rollouts
    # of 8 prompts x 8 samples, response length 8192, temperature 0.8, global batch 32, balanced.
    # Perf: tensor 1, pipeline 2, context 2, full uniform recompute, dynamic batch with 16384
    # tokens per GPU. PPO: PPO advantage estimator (creates the critic), critic learning rate
    # 1e-5, one critic-only warmup step with the actor frozen, normalized advantages, loss-level KL
    # (k1 type, zero coefficient) because reward-level KL is rejected with PPO, zero KL and entropy
    # coefficients, clip 0.2. Adam at 1e-6 constant. Sglang: one GPU per engine, memory fraction
    # 0.8, up to 512 running requests. Misc: no dropout, fp32 gradient all-reduce and softmax,
    # flash attention, one actor node, colocated. Submits with Megatron on the python path.
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    # --critic-load and --critic-lr fall back to --load and --lr. --critic-save falls back to
    # --save with a '_critic' suffix, a sibling dir so the two models keep separate iteration
    # trackers -- so none of the critic path flags need to be passed here.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}/ "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 300 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    ppo_args = (
        # Selecting ppo is what creates the critic; everything else here is tuning.
        "--advantage-estimator ppo "
        "--critic-lr 1e-5 "  # the critic usually wants a larger lr than the actor
        "--num-critic-only-steps 1 "  # value-function warmup: actor frozen for this many steps
        "--normalize-advantages "
        # Reward-level KL (--kl-coef) is rejected with ppo: the critic trains before the actor and
        # never sees ref log probs, so its value targets would silently omit that penalty. Use
        # loss-level KL instead.
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
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
        # Qwen3-4B fits on one GPU, so one engine per GPU beats sharding an engine across two.
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-max-running-requests 512 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
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
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command line entry: run prepare then execute.
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
