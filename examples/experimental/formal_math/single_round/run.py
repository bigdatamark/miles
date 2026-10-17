# [MODULE DOC]
# Launcher for single-round formal-math reinforcement learning (GRPO) on a Qwen3 model, intended to
# be run as a script. Module-level configuration is read from environment variables at import:
# dynamic sampling switch, reference and actor checkpoint paths, evaluation response length cap,
# dataset transform id (the name of a prepared dataset directory), and a mode (train, pass-at-k
# evaluation, or evaluation on a chunk of the training corpus; any other value fails an assertion).
# The model is fixed in the file (Qwen3-8B on eight GPUs). Two entry points: prepare downloads the
# Hugging Face checkpoint and, unless a reference checkpoint path was given, converts it to the
# Megatron distributed format; execute assembles the training argument string and submits the
# training job through the command utilities. Argument groups: checkpoint, rollout (custom reward
# function, 32 prompts times 8 samples, 8192 response tokens), evaluation (miniF2F and held-out
# corpus data), performance (tensor parallel 2, full recompute, dynamic batch size), GRPO, Adam
# optimiser with a constant 1e-6 learning rate, inference engine, and miscellaneous (colocated
# actor and rollout, rollout debug dumps under a run-id directory, pass-rate logging).
import os
from pathlib import Path


from miles.utils.external_utils import command_utils


# TODO unify "arg" prefix
enable_dynamic_sampling = bool(int(os.environ.get("ARG_ENABLE_DYNAMIC_SAMPLING", "0")))
arg_ref_load = os.environ.get("ARG_REF_LOAD")
arg_load = os.environ.get("ARG_LOAD")
eval_max_response_len = os.environ.get("ARG_EVAL_MAX_RESPONSE_LEN")

dataset_transform_id = os.environ.get("MILES_DATASET_TRANSFORM_ID")
mode = os.environ.get("MILES_MODE", "train")
assert mode in {"train", "eval_pass_at_k", "eval_flc"}

# MODEL_NAME, MODEL_TYPE = "Qwen3-4B", "qwen3-4B"
MODEL_NAME, MODEL_TYPE = "Qwen3-8B", "qwen3-8B"

NUM_GPUS = 8


def prepare():
    # [FUNCTION DOC] One-time preparation on the CPU side: creates the models and datasets
    # directories, downloads the model from the hub into the models directory, and, only if no
    # reference checkpoint path environment variable was set, converts the model to the Megatron
    # distributed checkpoint format with eight GPUs per node into the shared models directory.
    # Side effects: shell commands, network, disk.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    if arg_ref_load is None:
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=NUM_GPUS,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst="/root/models",
        )


def execute():
    # [FUNCTION DOC] Builds all training arguments and launches the job. A new run id is created
    # and used for the checkpoint directory (named by model, this script's name and run id), the
    # logging setup and the rollout debug dump path (which contains a rollout id placeholder filled
    # in later). Checkpoint load defaults: reference model from the converted checkpoint unless
    # overridden by environment; actor load from the override or the new save directory.
    # Mode handling: training uses 3000 rollouts; both evaluation modes use zero rollouts. Dynamic
    # sampling, if enabled, adds over-sampling of 64 prompts and a filter that keeps only groups
    # whose rewards have non-zero spread. Evaluation: every 20 steps, one sample per prompt, top-p
    # of one, response cap from the environment or 16384. In the chunk evaluation mode it evaluates
    # the training corpus slice named by an environment variable with 32 samples per prompt;
    # otherwise it evaluates miniF2F and the held-out corpus file, with 32 samples in pass-at-k
    # mode. All dataset paths contain the dataset transform id.
    # Finally runs the training with eight GPUs per node and the model type.
    # KNOWN ISSUE: the dataset transform id is not validated, so when its environment variable is
    # unset the dataset paths silently contain the text none. Also the one-engine-per-node setting
    # is a temporary workaround (see the inline comment) for an attention-library error.
    U = command_utils.default_config().create_backend()
    run_id: str = command_utils.create_run_id()

    load_save_path = f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}_{run_id}/"
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load {arg_ref_load or f'/root/models/{MODEL_NAME}_torch_dist'} "
        f"--load {arg_load or load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data /root/datasets/formal_math_single_round/{dataset_transform_id}/flc_train.jsonl "
        "--input-key prompt "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--custom-rm-path examples.experimental.formal_math.single_round.reward_fn.reward_fn "
        "--reward-key reward_value "
        "--log-reward-category reward_cat "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    if mode in {"eval_pass_at_k", "eval_flc"}:
        rollout_args += "--num-rollout 0 "
    else:
        rollout_args += "--num-rollout 3000 "

    if enable_dynamic_sampling:
        rollout_args += (
            "--over-sampling-batch-size 64 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = (
        "--eval-interval 20 "
        "--n-samples-per-eval-prompt 1 "
        f"--eval-max-response-len {eval_max_response_len or 16384} "
        "--eval-top-p 1 "
    )

    if mode == "eval_flc":
        flc_chunk = os.environ["MILES_FLC_CHUNK"]
        eval_args += (
            "--eval-prompt-data "
            f"flc /root/datasets/formal_math_single_round/{dataset_transform_id}/flc_train.jsonl@[{flc_chunk}] "
            # pass@32 is common for formal math
            "--n-samples-per-eval-prompt 32 "
        )
    else:
        eval_args += (
            "--eval-prompt-data "
            f"minif2f /root/datasets/formal_math_single_round/{dataset_transform_id}/minif2f_test.jsonl "
            f"flc /root/datasets/formal_math_single_round/{dataset_transform_id}/flc_test.jsonl "
        )
        if mode == "eval_pass_at_k":
            eval_args += "--n-samples-per-eval-prompt 32 "

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
        # see OOM when 9216 or 8192
        # TODO examine why this happens only in e.g. AC6588
        "--max-tokens-per-gpu 6144 "
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
        # "--rollout-num-gpus-per-engine 2 "
        "--rollout-num-gpus-per-engine 8 "  # temp use 1 engine per node to avoid flashinfer err
        "--sglang-mem-fraction-static 0.7 "
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
        f"--actor-num-nodes {os.environ.get('ARG_ACTOR_NUM_NODES', '1')} "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        # for debug
        f"--save-debug-rollout-data /root/shared_data/{run_id}/{{rollout_id}}.pt "
        "--log-passrate "
    )

    # should not use debug-rollout-only when doing eval, b/c the weights should be from megatron weights
    # if mode in {"eval_pass_at_k", "eval_flc"}:
    #     misc_args += "--debug-rollout-only "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    execute()
