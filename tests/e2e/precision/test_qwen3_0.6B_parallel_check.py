# [MODULE DOC] End-to-end precision test intended to show that training on a fixed batch gives the
# same gradient norm regardless of the parallelism layout, for Qwen3-0.6B on up to 8 GPUs,
# colocated. Flow per outer iteration: (1) run an 8-GPU training that dumps the rollout data and
# saves the gradient norm to files; (2) sweep over total GPU counts 8, 4 and 2 and over tensor,
# pipeline and context parallel sizes, each time re-running training that loads the dumped rollout
# and checks its gradient norm against the saved one (via the CI load-grad-norm flag). The second
# outer iteration repeats with per-token-loss calculation added to all arguments. Module state:
# an eval-enable flag read from an environment variable (default off, and unused in the arguments),
# model name/type, 8 GPUs. Registered for CUDA 8-GPU H100 (est. 2400 s) but disabled for bugs.
# Run as a script: prepare, strip proxy variables, then execute.
# KNOWN ISSUE: the sweep keeps a running count of remaining GPUs that is divided by the tensor
# size, then by the pipeline size, and is never restored between loop iterations, so after the
# first iterations the remaining count shrinks cumulatively (also as a float) and most parallel
# size combinations are skipped or wrong; it also never checks that the product of sizes divides
# the GPU count. The eval-enable flag is defined but never used. A fixme comment above the
# registration marks it as unfixed.
import os

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils

# FIXME: fix this
register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h100",
    labels=["megatron", "precision"],
    hardware=["hopper", "blackwell"],
    disabled="Disabled due to bugs.",
)

ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "0")))

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 8


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories, downloads Qwen3-0.6B and the dapo-math-17k
    # dataset, and converts the model to a Megatron distributed checkpoint under the models
    # directory on 8 GPUs. Side effects: shell commands, network, filesystem, GPU.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute():
    # [FUNCTION DOC] Builds the shared argument string and runs the reference-then-sweep loop twice.
    # Shared arguments: one rollout of 4 prompts times 8 samples, 8192 response tokens, temperature
    # 0.8, global batch 32; GRPO with k1 KL loss at zero coefficient and clip 4e-4; Adam at 1e-6;
    # SGLang with 2 GPUs per engine and mem fraction 0.8; CI test flag; dropout off; flash
    # attention; colocated, one actor node. Each outer iteration i (0 and 1): run a reference
    # training on 8 GPUs that saves rollout data to a file named from i and the rollout id
    # placeholder and saves the grad norm to a file named from i. Then for each GPU count in 8, 4, 2
    # and each tensor size in 1, 2, 4, 8, pipeline size in 1, 2, 4 and context size in 1, 2, 4, 8
    # that passes the remaining-GPU filters, run a training job loading that rollout data and grad
    # norm with those parallel sizes, sequence parallel, dynamic batching at 8192 tokens per GPU,
    # and that GPU count. After each outer iteration the per-token-loss flag is appended to the
    # shared arguments for later iterations. Returns nothing; failures come from the training
    # runs' own checks.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 2 " "--sglang-mem-fraction-static 0.8 "

    ci_args = "--ci-test "

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
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    for i in range(2):
        debug_data_path = f"data-{i}-{{rollout_id}}.pt"
        U.execute_train(
            train_args=train_args
            + (
                f"--save-debug-rollout-data {debug_data_path} "
                f"--ci-save-grad-norm grad_norms-{i}.pt "
                f"--actor-num-gpus-per-node {NUM_GPUS} "
            ),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )
        # 8 GPU CPU 1
        for num_gpus in [8, 4, 2]:
            remaining_gpus = num_gpus
            for tp_size in [1, 2, 4, 8]:
                remaining_gpus /= tp_size
                for pp_size in [1, 2, 4]:
                    if remaining_gpus < pp_size:
                        continue
                    remaining_gpus /= pp_size
                    for cp_size in [1, 2, 4, 8]:
                        if remaining_gpus < cp_size:
                            continue
                        args = train_args + (
                            f"--load-debug-rollout-data {debug_data_path} "
                            f"--ci-load-grad-norm grad_norms-{i}.pt "
                            f"--context-parallel-size {cp_size} "
                            f"--tensor-model-parallel-size {tp_size} "
                            f"--pipeline-model-parallel-size {pp_size} "
                            "--sequence-parallel "
                            f"--actor-num-gpus-per-node {num_gpus} "
                            "--use-dynamic-batch-size "
                            "--max-tokens-per-gpu 8192 "
                        )

                        U.execute_train(
                            train_args=args,
                            num_gpus_per_node=num_gpus,
                            megatron_model_type=MODEL_TYPE,
                        )
        train_args += "--calculate-per-token-loss "


if __name__ == "__main__":
    # TODO also use typer
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
