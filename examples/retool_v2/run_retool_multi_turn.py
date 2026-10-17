# [MODULE DOC] Launcher for multi-turn tool-use (code interpreter) GRPO training of Qwen3-4B on
# DAPO math, colocated on one or more nodes, with GPU count per node chosen from the detected
# or given hardware. Rollouts use the repo's multi-turn generate function with the tool specs,
# tool execution function and reward function from the sibling tool sandbox module, a Qwen 2.5
# style tool call parser and a configurable turn limit. The starting model is either a
# supervised-fine-tuned checkpoint (default, which also needs a larger rotary base) or the base
# model. Command line program; prepare downloads data and the model and converts the checkpoint;
# execute submits the job. Weights and Biases settings use a hard-coded project and group and
# the API key from the environment. Many paths are hard-coded under the root dir.
import os
from dataclasses import dataclass, field
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

WANDB_PROJECT = "miles-dev-retool-v2"
WANDB_GROUP = "sft-multi-turn-batch-32"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command line options extending the shared train config: mode (normal, or
    # debug_minimal: short responses, frequent saves, no eval), run id (generated per instance),
    # hardware (auto-detect, H100, GB200 or GB300), GPUs per node (default from hardware),
    # whether to start from the SFT model (default true), checkpoint save path, prompt data path,
    # maximum tool-use turns (16), GPUs per rollout engine (2) and extra arguments. Two fields
    # (HF checkpoint and reference load paths) are not user options; they are derived after
    # construction.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = field(default_factory=command_utils.create_run_id)
    hardware: Literal["auto", "H100", "GB200", "GB300"] = "auto"
    num_gpus_per_node: int | None = None
    use_sft_model: bool = True
    save_path: str = "/root/Qwen3-4B_miles/retool_v2_multi_turn"
    prompt_data: str = "/root/dapo-math-17k/dapo-math-17k.jsonl"
    generate_max_turns: int = 16
    rollout_num_gpus_per_engine: int = 2
    extra_args: str = ""

    # resolved in __post_init__, not set by user
    hf_checkpoint: str = field(init=False)
    ref_load: str = field(init=False)

    def __post_init__(self):
        # [FUNCTION DOC] Resolve derived settings: replace the hardware option with the resolved
        # hardware kind (detecting it when auto), default the GPUs per node to the standard count
        # for that hardware if not given (a given zero would also be replaced), and set the HF
        # checkpoint and reference load (converted checkpoint) paths either to the SFT model
        # folders under a font-info dir or to the base Qwen3-4B folders under the models dir.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.use_sft_model:
            self.hf_checkpoint = "/root/font-info/qwen3-4b-sft"
            self.ref_load = "/root/font-info/qwen3-4b-sft_torch_dist"
        else:
            self.hf_checkpoint = "/root/models/Qwen3-4B"
            self.ref_load = "/root/models/Qwen3-4B_torch_dist"


def _get_wandb_args() -> str:
    # [FUNCTION DOC] HELPER: Build the Weights and Biases flags with the module's fixed project
    # and group names and the API key read from the environment.
    # KNOWN ISSUE: if the key variable is unset the literal text for none is interpolated into
    # the key flag instead of omitting wandb, and this bypasses the shared default wandb helper.
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    return (
        "--use-wandb "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {WANDB_GROUP} "
        f"--wandb-key {WANDB_API_KEY} "
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Stage inputs under fixed root-dir paths: create folders and download the
    # DAPO math and AIME 2024 datasets. With the SFT model option, download the SFT model into the
    # derived HF checkpoint path and convert it to a distributed checkpoint under the font-info
    # dir (Qwen3-4B Megatron type, per-node GPU count). Otherwise download the base Qwen3-4B into
    # the models dir and convert it there. No return.
    # KNOWN ISSUE: dataset locations are fixed and not tied to the prompt data option, so a
    # custom prompt data path is never downloaded.
    U = args.create_backend()
    U.exec_command_cpu("mkdir -p /root/dapo-math-17k /root/aime-2024")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024")

    if args.use_sft_model:
        U.exec_command_cpu("mkdir -p /root/font-info")
        U.exec_command_cpu(f"hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir {args.hf_checkpoint}")
        U.convert_checkpoint(
            model_name="qwen3-4b-sft",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=args.num_gpus_per_node,
            hf_checkpoint=args.hf_checkpoint,
            dir_dst="/root/font-info",
        )
    else:
        U.exec_command_cpu("mkdir -p /root/models")
        U.exec_command_cpu("hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B")
        U.convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst="/root/models",
        )


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Build the training arguments and submit the job (Megatron model type fixed
    # to the Qwen3-4B type). Checkpoint: derived HF and reference paths, save path, save interval
    # (2 in debug mode else 1000) and, for the SFT model, a rotary base of 5000000. Custom: the
    # multi-turn generate function, tool specs and execution function from the sandbox module,
    # the Qwen 2.5 tool-call parser, the max turn count and multi-turn logging. Rollout: prompt
    # data with chat template and shuffle, the sandbox reward function, 3000 rollouts of 32
    # prompts x 8 samples, response length 100 in debug mode else 8192, temperature 1, global
    # batch 256, balanced. Eval (not in debug mode): every 20 rollouts on AIME, 16 samples,
    # response length 16384, top-p 1. GRPO with zero KL coefficient and clip 0.2/0.28; Adam 1e-6
    # constant; sglang with the configured GPUs per engine and memory fraction 0.7; perf: tensor
    # parallel 2 with sequence parallel, full recompute, dynamic batch of 9216 tokens per GPU.
    # Misc: node count from the base config, GPUs per node, colocated, no dropout, fp32 reductions,
    # flash attention and pass-rate logging. Submits with Megatron and the repo at fixed paths on
    # the python path.
    U = args.create_backend()
    megatron_model_type = "qwen3-4B"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 1000} "
        f"{'--rotary-base 5000000 ' if args.use_sft_model else ''}"
    )

    custom_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate "
        "--generate-tool-specs-path examples.retool_v2.tool_sandbox.tool_specs "
        "--generate-execute-tool-function-path examples.retool_v2.tool_sandbox.execute_tool "
        "--generate-tool-call-parser qwen25 "
        f"--generate-max-turns {args.generate_max_turns} "
        "--log-multi-turn "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--custom-rm-path examples.retool_v2.tool_sandbox.reward_func "
        "--reward-key score "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if args.mode != "debug_minimal":
        eval_args = (
            "--eval-interval 20 "
            "--eval-prompt-data aime /root/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
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
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} " "--sglang-mem-fraction-static 0.7 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--log-passrate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args()} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{custom_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        extra_env_vars={
            "PYTHONPATH": "/root/Megatron-LM/:/root/miles",
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command line entry: run prepare then execute.
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
