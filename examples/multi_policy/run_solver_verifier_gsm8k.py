# [MODULE DOC] Launcher for the solver/verifier multi-policy GSM8K recipe: two policies are
# trained in one run by the multi-policy train script, one trainer per policy with one GPU
# each, and one inference engine per policy. The solver (Qwen2.5-0.5B-Instruct by default)
# answers questions and the verifier (Qwen3-0.6B by default) judges the solver's answer; the
# custom generate function in this package produces both samples (see that module). Weights are
# loaded directly from HF checkpoints using the bridge conversion mode, so no checkpoint
# conversion is needed. Per-policy trainer and engine settings are passed as encoded pseudo-file
# YAML documents. The module is also reused by the split deployment launcher, which imports its
# argument builder and config helpers. Constants: policy ids solver and verifier (the solver is
# the leader whose model type is used for the job), the eval dataset name, the train script name,
# shared per-trainer hyperparameter overrides (constant LR schedule, weight decay 0.1, Adam betas
# 0.9/0.98, zero KL and entropy coefficients, KL type low variance, clip 0.2/0.28).
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import compute_model_args_overrides, encode_pseudo_file

SOLVER_MODEL_ID: str = "solver"
VERIFIER_MODEL_ID: str = "verifier"
MODEL_IDS: list[str] = [SOLVER_MODEL_ID, VERIFIER_MODEL_ID]
LEADER_MODEL_ID: str = MODEL_IDS[0]
EVAL_DATASET_NAME: str = "gsm8k"
TRAIN_SCRIPT: str = "train_multi_policy.py"
TRAIN_EXTRA_ENV_VARS: dict[str, str] = {}

# Hyperparameters roughly follow test_qwen2.5_0.5B_gsm8k_async.py.
SHARED_TRAINER_OVERRIDES = dict(
    lr_decay_style="constant",
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    entropy_coef=0.0,
    eps_clip=0.2,
    eps_clip_high=0.28,
)


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command line options extending the shared train config: number of rollouts (3),
    # checkpoint save interval (20), GPUs per node (4), the two HF model names and Megatron model
    # types, rollout GPUs per policy (1), actor GPUs per policy (1), data and model dirs and
    # free-form extra arguments. With the defaults the node's 4 GPUs are 2 trainer GPUs and 2
    # engine GPUs.
    num_rollout: int = 3
    save_interval: int = 20
    num_gpus_per_node: int = 4
    solver_model_name: str = "Qwen2.5-0.5B-Instruct"
    verifier_model_name: str = "Qwen3-0.6B"
    solver_megatron_model_type: str = "qwen2.5-0.5B"
    verifier_megatron_model_type: str = "qwen3-0.6B"
    rollout_num_gpus_per_model: int = 1
    actor_num_gpus_per_policy: int = 1
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    extra_args: str = ""

    @property
    def rollout_num_gpus(self) -> int:
        # [FUNCTION DOC] Total rollout GPUs: GPUs per policy times the number of policies (2).
        return self.rollout_num_gpus_per_model * len(MODEL_IDS)

    @property
    def model_path_of_model_id(self) -> dict[str, str]:
        # [FUNCTION DOC] Mapping from policy id to the local HF model folder: the model dir joined
        # with the solver or verifier model name.
        return {
            SOLVER_MODEL_ID: f"{self.model_dir}/{self.solver_model_name}",
            VERIFIER_MODEL_ID: f"{self.model_dir}/{self.verifier_model_name}",
        }

    @property
    def megatron_model_type_of_model_id(self) -> dict[str, str]:
        # [FUNCTION DOC] Mapping from policy id to its Megatron model type string.
        return {
            SOLVER_MODEL_ID: self.solver_megatron_model_type,
            VERIFIER_MODEL_ID: self.verifier_megatron_model_type,
        }


def prepare(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Stage inputs: create the model and data dirs, download both HF models into
    # their folders and the GSM8K dataset. No checkpoint conversion (bridge mode loads HF
    # directly). No return.
    U = args.create_backend()
    model_path_of_model_id = args.model_path_of_model_id
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download Qwen/{args.solver_model_name} --local-dir {model_path_of_model_id[SOLVER_MODEL_ID]}"
    )
    U.exec_command_cpu(
        f"hf download Qwen/{args.verifier_model_name} --local-dir {model_path_of_model_id[VERIFIER_MODEL_ID]}"
    )
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def execute(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Build the full argument string with default settings and submit it.
    launch_train(build_train_args(args), args)


def build_train_args(
    args: ScriptArgs,
    *,
    wandb_args: str | None = None,
    megatron_config: dict | None = None,
    sglang_config: dict | None = None,
    rollout_num_gpus: int | None = None,
) -> str:
    # [FUNCTION DOC] Build the training argument string. Optional overrides (used by the split
    # deployment launcher to build per-part arguments): wandb args text, a trainer configuration
    # mapping, an engine configuration mapping and a rollout GPU count; each override is used when
    # given and non-empty, else the default computed for all policies is used (an empty mapping
    # counts as not given; a rollout count of zero is honored because only none is tested).
    # Checkpoint: HF checkpoint and reference load both from the solver's folder (policy-specific
    # paths come from the trainer config), a save dir and save interval. Policy: the trainer and
    # engine configs as encoded pseudo-files of YAML, and the custom generate function path
    # of this package. Rollout: fully async, GSM8K train parquet with messages key, shuffle,
    # rollout count, 32 prompts x 8 samples, response length 1024, temperature 1, global batch
    # 256, a dynamic sampling filter keeping only groups whose rewards are not all equal, reward
    # key and category logging, and retract pause mode. Eval: every 20 rollouts on the GSM8K test
    # set, one sample, response length 1024, top-k 1, with the per-policy eval split logging hook.
    # Perf: all parallel sizes 1, dynamic batch 9216 tokens per GPU. GRPO with KL loss; Adam with
    # learning rate 1e-6; sglang one GPU per engine, memory fraction 0.65, metrics on. CI: test
    # flag plus debug event data dir and a debug rollout data path template. Misc: no dropout, fp32
    # gradient all-reduce and softmax, flash attention, one actor node with the per-policy actor
    # GPU count, rollout GPUs (override or default), bridge conversion mode. Final string joins
    # all parts, the wandb text (override, else default args keyed to this file and run id) and the
    # extra args.
    # Note: the per-engine GPU count is written both as a fixed 1 here and as a fixed 1 inside
    # the engine config entries, so the two must be kept in step by hand.
    events_dir = compute_events_dir(args)
    solver_path = args.model_path_of_model_id[SOLVER_MODEL_ID]

    ckpt_args = (
        f"--hf-checkpoint {solver_path}/ "
        f"--ref-load {solver_path}/ "
        f"--save {compute_save_dir(args)} "
        f"--save-interval {args.save_interval} "
    )

    policy_args = (
        f"--megatron-config {encode_pseudo_file(yaml.dump(megatron_config or compute_megatron_config(args)))} "
        f"--sglang-config {encode_pseudo_file(yaml.dump(sglang_config or compute_sglang_config(args)))} "
        "--custom-generate-function-path examples.multi_policy.solver_verifier.generate "
    )

    rollout_args = (
        "--fully-async "
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--reward-key reward_value "
        "--log-reward-category outcome "
        "--pause-generation-mode retract "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data {EVAL_DATASET_NAME} {args.data_dir}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
        "--custom-eval-rollout-log-function-path examples.multi_policy.solver_verifier.split_eval_data_by_policy "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = "--advantage-estimator grpo " "--use-kl-loss "

    optimizer_args = "--optimizer adam " "--lr 1e-6 "

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.65 " "--sglang-enable-metrics "

    ci_args = (
        "--ci-test "
        f"--save-debug-event-data {events_dir} "
        f"--save-debug-rollout-data {compute_rollout_data_path_template(args)} "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_policy} "
        f"--rollout-num-gpus {args.rollout_num_gpus if rollout_num_gpus is None else rollout_num_gpus} "
        "--megatron-to-hf-mode bridge "
    )

    return (
        f"{ckpt_args} "
        f"{policy_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args if wandb_args is not None else command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{eval_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )


def launch_train(train_args: str, args: ScriptArgs) -> None:
    # [FUNCTION DOC] Submit the given argument string via the backend using the multi-policy
    # train script, the node GPU count, the leader (solver) policy's Megatron model type and a
    # copy of the extra environment variable mapping (currently empty). No return.
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type_of_model_id[LEADER_MODEL_ID],
        train_script=TRAIN_SCRIPT,
        extra_env_vars=dict(TRAIN_EXTRA_ENV_VARS),
    )


def compute_megatron_config(args: ScriptArgs, *, model_ids: list[str] | None = None) -> dict:
    # [FUNCTION DOC] Build the trainers configuration: a mapping with one key holding a list of
    # per-policy trainer entries for the given policy ids (all policies in order solver, verifier
    # when none or empty).
    return dict(trainers=[_compute_trainer_config(args, model_id) for model_id in model_ids or MODEL_IDS])


def compute_sglang_config(args: ScriptArgs, *, model_ids: list[str] | None = None) -> dict:
    # [FUNCTION DOC] Build the inference engines configuration: a mapping with one key holding a
    # list of per-policy engine entries for the given policy ids (all policies when none or empty).
    return dict(sglang=[_compute_sglang_model_config(args, model_id) for model_id in model_ids or MODEL_IDS])


def compute_trainer_id(model_id: str) -> str:
    # [FUNCTION DOC] Trainer id of a policy: the policy id, a hyphen, then the actor role name.
    return f"{model_id}-{ACTOR_ROLE}"


def compute_output_base_dir(config: ExecuteTrainConfig) -> Path:
    # [FUNCTION DOC] Root of this run's outputs: the config output dir, a fixed example folder
    # name, then the run id. Returns a path.
    return Path(config.output_dir) / "multi_policy_solver_verifier" / config.run_id


def compute_events_dir(config: ExecuteTrainConfig) -> Path:
    # [FUNCTION DOC] Folder named events under the run's output root (debug event data).
    return compute_output_base_dir(config) / "events"


def compute_rollout_data_path_template(config: ExecuteTrainConfig) -> str:
    # [FUNCTION DOC] Path template, as a string, for per-rollout debug data files: the rollout
    # data folder under the run's output root with a file name made of a literal rollout id
    # placeholder (filled in later by the trainer) and the torch file extension.
    return str(compute_output_base_dir(config) / "rollout_data" / "{rollout_id}.pt")


def compute_save_dir(config: ExecuteTrainConfig) -> Path:
    # [FUNCTION DOC] Checkpoint folder under the run's output root.
    return compute_output_base_dir(config) / "ckpt"


def _compute_trainer_config(args: ScriptArgs, model_id: str) -> dict:
    # [FUNCTION DOC] HELPER: Trainer entry for one policy: the policy id, its trainer id and an
    # overrides mapping combining the HF checkpoint and reference load (both the policy's model
    # folder), the model architecture overrides derived from its Megatron model type, and the
    # shared hyperparameter overrides.
    model_path = args.model_path_of_model_id[model_id]
    return dict(
        model_id=model_id,
        trainer_id=compute_trainer_id(model_id),
        overrides=dict(
            hf_checkpoint=model_path,
            ref_load=model_path,
            **compute_model_args_overrides(args.megatron_model_type_of_model_id[model_id]),
            **SHARED_TRAINER_OVERRIDES,
        ),
    )


def _compute_sglang_model_config(args: ScriptArgs, model_id: str) -> dict:
    # [FUNCTION DOC] HELPER: Engine entry for one policy: named by the policy id, serving the
    # policy's model folder, receiving weight updates from training, one GPU per engine, and a
    # single regular server group with the per-policy rollout GPU count.
    return dict(
        name=model_id,
        model_path=args.model_path_of_model_id[model_id],
        update_weights=True,
        num_gpus_per_engine=1,
        server_groups=[dict(worker_type="regular", num_gpus=args.rollout_num_gpus_per_model)],
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Command line entry: run prepare then execute.
    prepare(args)
    execute(args)


# TODO: unify this launcher when the example scripts are refactored
if __name__ == "__main__":
    typer.run(main)
