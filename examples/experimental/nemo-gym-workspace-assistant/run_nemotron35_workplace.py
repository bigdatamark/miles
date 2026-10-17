# [MODULE DOC]
# Launcher for asynchronous GRPO training of a Nemotron model on the Workplace Assistant tool-use
# tasks across two eight-GPU nodes (one for training actors, the rest of the cluster for rollout
# inference), run as a script with a JSON configuration file. The configuration keys fill the
# script argument record, which validates the topology and batch arithmetic. The launcher builds
# the training argument string from helper functions that render ordered mappings to command line
# flags, grouped as checkpoint, rollout, learning (parallelism, algorithm, optimiser), serving
# (inference engine, routing replay, session server with token-in-token-out handling) and
# miscellaneous (dropout, precision, tracing, metrics, a training pre-step hook). Environment
# forwarded to the job: the verifier resource URL, Hugging Face cache locations, and a module
# search path entry for this example directory. Submitted with the asynchronous training script.
"""Nemotron 3.5 Lightning Workplace Assistant GRPO with separate rollout and training nodes.

Requires two joined eight-GPU Ray nodes, matching model and dataset paths on both,
and the NeMo Gym Workplace Assistant verifier. The model starts from original HF
weights. The native resource service owns tool state and binary reward; all policy
turns use Miles TITO sessions.

Args:
    --config: JSON object containing the ScriptArgs fields below.

Example:
    python examples/experimental/nemo-gym-workspace-assistant/run_nemotron35_workplace.py \
        --config /path/to/launcher_config.json
"""

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from tap import Tap

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # [CLASS DOC]
    # Validated launcher configuration (extends the shared execute-train configuration). Fields and
    # defaults: run id (generated per instance), nodes 2, GPUs per node 8, model and data
    # directories, model name, data file name, Megatron path, verifier URL (local port 8211), Hugging
    # Face cache directory, learning rate 3e-7, rollout batch 8 prompts, group size 16, global batch
    # 128, 1000 rollouts, save interval 100, response length 65536, context length 81920, maximum
    # tokens per GPU 16384, and pause generation mode (only retract is accepted).
    run_id: str = field(default_factory=U.create_run_id)
    num_nodes: int = 2
    num_gpus_per_node: int = 8
    model_dir: str = "/root/models"
    model_name: str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    data_dir: str = "/root/datasets"
    data_file: str = "workplace_train.jsonl"
    megatron_path: str = "/root/Megatron-LM"
    verifier_url: str = "http://127.0.0.1:8211"
    hf_cache_dir: str = "/root/.cache/huggingface"
    learning_rate: float = 3e-7
    rollout_batch_size: int = 8
    group_size: int = 16
    global_batch_size: int = 128
    num_rollout: int = 1000
    save_interval: int = 100
    response_length: int = 65536
    context_length: int = 81920
    max_tokens_per_gpu: int = 16384
    pause_generation_mode: str = "retract"

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validation after construction; raises a value error unless: the node and GPU
        # counts are exactly two and eight; global batch size equals rollout batch size times group
        # size; response length is positive and strictly below context length; the batch sizes, group
        # size, rollout count, save interval and token budget are all positive; and the pause
        # generation mode is retract (needed by routing replay).
        if (self.num_nodes, self.num_gpus_per_node) != (2, 8):
            raise ValueError("This example requires two nodes with eight GPUs each")
        if self.global_batch_size != self.rollout_batch_size * self.group_size:
            raise ValueError("global_batch_size must equal rollout_batch_size * group_size")
        if not 0 < self.response_length < self.context_length:
            raise ValueError("response_length must be positive and smaller than context_length")
        if (
            min(
                self.global_batch_size,
                self.rollout_batch_size,
                self.group_size,
                self.num_rollout,
                self.save_interval,
                self.max_tokens_per_gpu,
            )
            <= 0
        ):
            raise ValueError("Batch sizes, update counts and token budgets must be positive")
        if self.pause_generation_mode != "retract":
            raise ValueError("This routing-replay recipe requires pause_generation_mode=retract")

    @property
    def run_name(self) -> str:
        # [FUNCTION DOC] Property: descriptive run name joining the run id, a fixed model/topology
        # label, the global batch size, group size and pause mode; used as the logging run id.
        return f"{self.run_id}-nemotron35-lightning-workplace-async-2n-bs{self.global_batch_size}-g{self.group_size}-{self.pause_generation_mode}"


def _flags(values: dict[str, object]) -> str:
    # [FUNCTION DOC] HELPER: Renders a mapping to a shell-quoted command line string in mapping
    # order. Entries whose value is false or none are skipped. Each other key becomes a flag with a
    # double dash prefix and underscores replaced by hyphens; the value follows as text unless the
    # value is exactly true (a bare switch). Zero and empty text are emitted as values.
    tokens = []
    for key, value in values.items():
        if value is False or value is None:
            continue
        tokens.append("--" + key.replace("_", "-"))
        if value is not True:
            tokens.append(str(value))
    return shlex.join(tokens)


def _learning_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Learning-related flags in three groups joined by spaces. Performance:
    # tensor parallel 2 with sequence parallel, pipeline parallel 2, context parallel 1, expert
    # parallel 2, expert tensor parallel 1, full uniform recompute of one layer, dynamic batch size
    # with the token budget per GPU, log-prob chunk size 128, sequence length equal to the context
    # length, and CPU optimiser offload with overlapped transfers and precision-aware optimiser.
    # Algorithm: GRPO with KL loss enabled at zero coefficient (low-variance type), zero entropy
    # coefficient, clip range 0.2 below and 0.28 above, use of rollout log probabilities, and the
    # pause generation mode. Optimiser: Adam, configured learning rate, constant schedule, weight decay
    # 0.1, betas 0.9 and 0.98.
    performance_args = _flags(
        {
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 2,
            "expert_tensor_parallel_size": 1,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "use_dynamic_batch_size": True,
            "max_tokens_per_gpu": args.max_tokens_per_gpu,
            "log_probs_chunk_size": 128,
            "seq_length": args.context_length,
            "optimizer_cpu_offload": True,
            "overlap_cpu_optimizer_d2h_h2d": True,
            "use_precision_aware_optimizer": True,
        }
    )
    algorithm_args = _flags(
        {
            "advantage_estimator": "grpo",
            "use_kl_loss": True,
            "kl_loss_coef": 0,
            "kl_loss_type": "low_var_kl",
            "entropy_coef": 0,
            "eps_clip": 0.2,
            "eps_clip_high": 0.28,
            "use_rollout_logprobs": True,
            "pause_generation_mode": args.pause_generation_mode,
        }
    )
    optimizer_args = _flags(
        {
            "optimizer": "adam",
            "lr": args.learning_rate,
            "lr_decay_style": "constant",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
        }
    )
    return " ".join([performance_args, algorithm_args, optimizer_args])


def _checkpoint_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Checkpoint flags. The Hugging Face checkpoint and reference load both
    # point to the model directory joined with the model name (training starts from the original
    # weights); save directory is a checkpoints sub-directory of the configured output directory,
    # with the configured interval; weights are converted through the bridge mode; optimiser and
    # random-state loading are disabled and fine-tune mode is on.
    checkpoint = str(Path(args.model_dir) / args.model_name)
    return _flags(
        {
            "hf_checkpoint": checkpoint,
            "ref_load": checkpoint,
            "save": str(Path(args.output_dir) / "checkpoints"),
            "save_interval": args.save_interval,
            "megatron_to_hf_mode": "bridge",
            "no_load_optim": True,
            "no_load_rng": True,
            "finetune": True,
        }
    )


def _rollout_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Rollout flags. Prompt data is the data directory joined with the data
    # file; input, metadata and label keys are prompt, metadata and label; the chat template is not
    # applied (the flag value false is omitted); custom generate, agent and reward function paths
    # point at modules in this example directory; maximum sequence length equals the context length;
    # the dynamic sampling filter removes aborted samples; shuffle on; reward type deepscaler (with
    # the custom reward path configured as well); the rollout count, rollout batch size, group size
    # (samples per prompt), global batch size, response length cap, temperature 1 and data balancing.
    return _flags(
        {
            "prompt_data": str(Path(args.data_dir) / args.data_file),
            "input_key": "prompt",
            "metadata_key": "metadata",
            "label_key": "label",
            "apply_chat_template": False,
            "custom_generate_function_path": "workplace_generate.generate",
            "custom_agent_function_path": "workplace_agent.run",
            "max_seq_len": args.context_length,
            "dynamic_sampling_filter_path": "miles.rollout.filter_hub.common_filters.apply_aborted_filter",
            "rollout_shuffle": True,
            "rm_type": "deepscaler",
            "custom_rm_path": "workplace_generate.reward_func",
            "num_rollout": args.num_rollout,
            "rollout_batch_size": args.rollout_batch_size,
            "n_samples_per_prompt": args.group_size,
            "global_batch_size": args.global_batch_size,
            "rollout_max_response_len": args.response_length,
            "rollout_temperature": 1,
            "balance_data": True,
        }
    )


def _serving_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Inference serving flags: one GPU per engine, static memory fraction
    # 0.7, engine context length equal to the configured context length, routing replay on, round
    # robin router policy, engine metrics on, the version-one session server with 8 workers listening
    # on all interfaces at port 31000 and pinned to the head node's rollout manager, the nemotron3
    # token handling model, the nemotron_3 reasoning parser and the qwen3_coder tool call parser.
    return _flags(
        {
            "rollout_num_gpus_per_engine": 1,
            "sglang_mem_fraction_static": 0.7,
            "sglang_context_length": args.context_length,
            "use_rollout_routing_replay": True,
            "sglang_router_policy": "round_robin",
            "sglang_enable_metrics": True,
            "use_session_server": "v1",
            "session_server_workers": 8,
            "pin_rollout_manager_to_head": True,
            "session_server_ip": "0.0.0.0",
            "session_server_port": 31000,
            "tito_model": "nemotron3",
            "sglang_reasoning_parser": "nemotron_3",
            "sglang_tool_call_parser": "qwen3_coder",
        }
    )


def _miscellaneous_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Remaining flags: zero dropouts; fp32 gradient accumulation and
    # attention softmax; automatic attention backend; one actor node and the GPU counts per node
    # (rollout GPU count equal to GPUs per node, no colocation); zero multi-token-prediction loss
    # scale; a pre-train-step hook from this example's training checks module; detail dumps into a
    # traces directory under the output directory; dashboard, training and rollout entropy
    # observation; and Prometheus metrics on port 9090 forwarded to the dashboard.
    return _flags(
        {
            "attention_dropout": 0,
            "hidden_dropout": 0,
            "accumulate_allreduce_grads_in_fp32": True,
            "attention_softmax_in_fp32": True,
            "attention_backend": "auto",
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": args.num_gpus_per_node,
            "num_gpus_per_node": args.num_gpus_per_node,
            "rollout_num_gpus": args.num_gpus_per_node,
            "mtp_loss_scaling_factor": 0,
            "custom_megatron_before_train_step_hook_path": "workplace_training_checks.before_train_step",
            "dump_details": str(Path(args.output_dir) / "traces"),
            "use_miles_dashboard": True,
            "observe_training_entropy": True,
            "use_rollout_entropy": True,
            "use_prometheus": True,
            "prometheus_port": 9090,
            "dashboard_forward_prometheus": True,
        }
    )


def execute(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Submits the training job. Concatenates the checkpoint, rollout, learning,
    # serving and miscellaneous flag strings and the default logging arguments (run name as run
    # id) with spaces, and submits through the configuration's backend using the asynchronous
    # training script, the configured GPUs per node, the nemotron-3 nano 30B model type and the
    # Megatron path. Extra environment: an empty value for the keep-MTP switch, the verifier
    # resource URL, Hugging Face home and hub cache locations, and a module search path of this
    # example directory under the repository base.
    args.create_backend().execute_train(
        train_args=" ".join(
            [
                _checkpoint_args(args),
                _rollout_args(args),
                _learning_args(args),
                _serving_args(args),
                _miscellaneous_args(args),
                U.get_default_wandb_args(__file__, run_id=args.run_name),
            ]
        ),
        config=args,
        train_script="train_async.py",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="nemotron-3-nano-30b-a3b",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "MILES_NEMOTRONH_KEEP_MTP": "",
            "WORKPLACE_RESOURCE_URL": args.verifier_url,
            "HF_HOME": args.hf_cache_dir,
            "HUGGINGFACE_HUB_CACHE": str(Path(args.hf_cache_dir) / "hub"),
            "PYTHONPATH": str(Path(U.repo_base_dir) / "examples/experimental/nemo-gym-workspace-assistant"),
        },
    )


class _CLI(Tap):
    # [CLASS DOC] HELPER: Command line schema with one required option, the path of the JSON
    # configuration file.
    config: Path


def main() -> None:
    # [FUNCTION DOC] Entry point: parses the command line, reads the JSON configuration file,
    # builds the validated script arguments from its keys (unknown keys raise a type error,
    # invalid values a value error) and runs the launcher.
    cli = _CLI().parse_args()
    execute(ScriptArgs(**json.loads(cli.config.read_text())))


if __name__ == "__main__":
    main()
