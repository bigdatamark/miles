# [MODULE DOC] Launcher for a split deployment of a small Qwen3-0.6B GRPO run: instead of one
# job, the run is installed as several separate deployments (one trainer, one inference
# deployment per engine, and one primary deployment that coordinates them). Each launch command
# installs exactly one part, chosen by the deploy-component option inherited from the shared
# train config (plus a deploy instance id for inference parts, named by engine index), and all
# parts must be given the same run id, namespace and run uuid so that the address book can
# compute consistent in-cluster addresses. Every command also runs the prepare step and then
# submits through the backend. Role-specific arguments are appended to a common argument string:
# the shared object store arguments for all roles, the inference controller address for
# inference parts, and the trainer controller addresses plus expected cell count for the primary.
from dataclasses import dataclass

import typer
from examples.infra_features.split_deployment.address_book import (
    DEFAULT_TRAINER_ID,
    RunAddressBook,
    init_expected_num_cells_arg,
)

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import DeployComponent


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command line options extending the shared train config (which supplies the
    # deploy component, deploy instance id, namespace, run id and run uuid): model name and
    # Megatron type, number of rollouts (3), actor GPU count (4), number of inference engines
    # (2), GPUs per engine (1), data/model/Megatron directories and free-form extra arguments.
    model_name: str = "Qwen3-0.6B"
    megatron_model_type: str = "qwen3-0.6B"
    num_rollout: int = 3
    actor_num_gpus: int = 4
    num_engines: int = 2
    gpus_per_engine: int = 1
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    @property
    def num_gpus_per_node(self) -> int:
        # [FUNCTION DOC] Total GPUs of the whole run: actor GPUs plus engines times GPUs per engine.
        return self.actor_num_gpus + self.num_engines * self.gpus_per_engine


def prepare(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Stage inputs: create directories, download the HF model and the GSM8K
    # dataset, and convert the HF checkpoint to a distributed Megatron checkpoint using one
    # process per actor GPU. Runs on every part's launch command. No return.
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


def build_deployment_train_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] Build the full argument string for the part selected by the deploy component.
    # Builds the run's address book from the config (assertion errors if namespace or run uuid are
    # missing) and the shared object store arguments. Trainer part: common arguments sized for the
    # rollout GPUs of the whole run, plus the shared arguments. Inference part: common arguments
    # sized for one engine's GPUs, plus shared arguments and the inference controller address.
    # Primary part: whole-run sizing, shared arguments, the trainer controller address for the
    # default trainer id (no instance id) and the expected cell count equal to the number of
    # engines. Any other component raises an assertion error that lists the parts to launch, one
    # command at a time. Returns the string.
    address_book = RunAddressBook.of_config(args)
    shared = address_book.shared_object_store_args()
    whole_run_rollout_num_gpus = args.num_engines * args.gpus_per_engine

    match args.deploy_component:
        case DeployComponent.TRAINER:
            return build_train_args(args, rollout_num_gpus=whole_run_rollout_num_gpus) + shared
        case DeployComponent.INFERENCE:
            return (
                build_train_args(args, rollout_num_gpus=args.gpus_per_engine)
                + shared
                + address_book.inference_controller_addr_arg()
            )
        case DeployComponent.PRIMARY:
            return (
                build_train_args(args, rollout_num_gpus=whole_run_rollout_num_gpus)
                + shared
                + address_book.trainer_controller_addrs_arg(
                    deploy_instance_id_of_trainer_id={DEFAULT_TRAINER_ID: None}
                )
                + init_expected_num_cells_arg(args.num_engines)
            )
        case _:
            raise AssertionError(
                f"one command of this example installs one part of the run, and --deploy-component "
                f"{args.deploy_component.value} names no part; launch "
                f"{[one.value for one, _ in compute_deployment_identities(args)]} one command at a time"
            )


def compute_deployment_identities(args: ScriptArgs) -> list[tuple[DeployComponent, str | None]]:
    # [FUNCTION DOC] List the parts that make up the run as pairs of component and instance id:
    # the trainer (no id), one inference part per engine with ids numbered from zero prefixed by
    # a letter, and the primary (no id), in that order. Used for error messages and for tooling
    # that launches the parts.
    return [
        (DeployComponent.TRAINER, None),
        *[(DeployComponent.INFERENCE, f"e{index}") for index in range(args.num_engines)],
        (DeployComponent.PRIMARY, None),
    ]


def build_train_args(args: ScriptArgs, *, rollout_num_gpus: int) -> str:
    # [FUNCTION DOC] Common training argument string shared by all parts; only the rollout GPU
    # count varies by part. Checkpoint: HF model and reference load from the converted checkpoint
    # (no save or load of training state). Rollout: GSM8K parquet, messages key, chat template,
    # shuffle, deterministic random reward, response length 200, temperature 0.8, 32 prompts x 8
    # samples, rollout count from options, the given rollout GPU count and GPUs per engine.
    # Optimizer: Adam 1e-6 constant, weight decay 0.1, betas 0.9/0.98, fp32 gradient all-reduce.
    # Misc: no dropout, flash attention, context parallel 2, one actor node with the actor GPU
    # count, global batch 256, dynamic batch with 32768 tokens per GPU, GRPO with clip 0.2.
    # Then default wandb args and the free-form extra args. Parts joined by single spaces.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
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
        f"--rollout-num-gpus {rollout_num_gpus} "
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
        f"{args.extra_args} "
    )


def launch_train(train_args: str, args: ScriptArgs) -> None:
    # [FUNCTION DOC] Submit the given argument string through the backend with the whole-run
    # GPU count and model type; default train script. No return.
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Command line entry: build the part's arguments first (so a bad component
    # fails before any download), then prepare the inputs, then launch the training submission.
    train_args = build_deployment_train_args(args)
    prepare(args)
    launch_train(train_args, args)


# TODO: unify this launcher when the example scripts are refactored
if __name__ == "__main__":
    typer.run(main)
