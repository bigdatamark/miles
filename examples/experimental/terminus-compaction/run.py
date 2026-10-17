# [MODULE DOC] Self-executing launcher for GRPO training of GLM-4.7-Flash on Terminal-Bench 2 with
# the Harbor Terminus 2 agent and trajectory compaction, on one colocated 8-GPU node. Reuses the
# agent function, reward function and rollout class of the swe-agent-harbor-docker example (that
# directory is put on the python path of the job) and enables session server v2 so every
# summarization-created trajectory segment is kept. The argument string is assembled from small
# private builder functions, one per block; main optionally converts the checkpoint, then submits
# the training job. Paths default under the model dir and the output dir/run id.
"""Train GLM-4.7-Flash with Terminus 2 trajectory compaction.

This one-node recipe uses Harbor for Terminal-Bench 2 environments and Miles'
session server v2 to retain the trajectory segments created by Terminus 2
summarization. Start the Harbor agent server as described in the adjacent
README before launching this script.

The default recipe runs 100 GRPO steps over all 89 Terminal-Bench 2 tasks, with
four prompts and eight independent rollouts per prompt in each step.

Example:
    python examples/experimental/terminus-compaction/run.py \
        --skip-prepare \
        --model-dir /path/to/models \
        --output-dir /path/to/output \
        --agent-server-url http://agent-server.example:11000 \
        --session-server-ip 0.0.0.0 \
        --session-server-external-host trainer.example
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

_HARBOR_PIPELINE_DIR = command_utils.repo_base_dir / "examples" / "swe-agent-harbor-docker"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Launcher configuration (extends the shared train config). Groups: mode (normal or
    # rollout-only debug), run id (generated per instance), model identity (glm4.7-flash, 8 GPUs
    # per node); paths (skip-prepare flag, model dir, model name, and optional overrides for HF
    # checkpoint, reference load, save dir and trace dump dir, all derived when empty; prompt data
    # file listing the 89 tasks); training scale (max sequence length 32768, response length 8192,
    # 100 rollouts, 4 prompts per step, 8 samples per prompt, global batch 32, save interval 100);
    # agent settings (server url from AGENT_SERVER_URL else http://127.0.0.1:11000, model name from
    # AGENT_MODEL_NAME else "model", trial timeout 7200 s, session server external host, host ip
    # from MILES_HOST_IP, session server bind ip from MILES_SESSION_SERVER_IP); and Prometheus
    # settings (enabled, port 9090). Environment defaults are read when an instance is created.
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = field(default_factory=command_utils.create_run_id)
    megatron_model_type: str = "glm4.7-flash"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    skip_prepare: bool = False
    model_dir: str = "/root/models"
    model_name: str = "GLM-4.7-Flash"
    hf_checkpoint: str = ""
    ref_load: str = ""
    save_dir: str = ""
    save_traces_dir: str = ""
    prompt_data: str = "/root/tb2_train_89.jsonl"

    max_seq_len: int = 32768
    rollout_max_response_len: int = 8192
    num_rollout: int = 100
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 100

    agent_server_url: str = field(default_factory=lambda: os.environ.get("AGENT_SERVER_URL", "http://127.0.0.1:11000"))
    agent_model_name: str = field(default_factory=lambda: os.environ.get("AGENT_MODEL_NAME", "model"))
    agent_trial_timeout: int = 7200
    # The host agents outside the cluster reach every session server on; passed as
    # --session-server-external-host, which also keeps the session servers on the head node.
    session_server_external_host: str = ""
    miles_host_ip: str = field(default_factory=lambda: os.environ.get("MILES_HOST_IP", ""))
    session_server_ip: str = field(default_factory=lambda: os.environ.get("MILES_SESSION_SERVER_IP", ""))

    use_prometheus: bool = True
    prometheus_port: int = 9090


def _hf_checkpoint(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the explicit HF checkpoint path, else model dir joined with the
    # model name.
    return args.hf_checkpoint or str(Path(args.model_dir) / args.model_name)


def _ref_load(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the explicit reference load path, else model dir joined with
    # "<model name>_torch_dist".
    return args.ref_load or str(Path(args.model_dir) / f"{args.model_name}_torch_dist")


def _save_dir(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the explicit save dir, else <output dir>/<run id>/checkpoints.
    return args.save_dir or str(Path(args.output_dir) / args.run_id / "checkpoints")


def _save_traces_dir(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the explicit trace dump dir, else <output dir>/<run id>/details.
    return args.save_traces_dir or str(Path(args.output_dir) / args.run_id / "details")


def _checkpoint_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the checkpoint flags: HF checkpoint, reference load, save path
    # and save interval.
    return (
        f"--hf-checkpoint {_hf_checkpoint(args)} "
        f"--ref-load {_ref_load(args)} "
        f"--save {_save_dir(args)} "
        f"--save-interval {args.save_interval} "
    )


def _rollout_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the rollout flags: prompt data with input key prompt and
    # metadata key metadata, shuffle, rollout count, rollout batch size, samples per prompt,
    # temperature 0.8 (fixed), response length limit, max sequence length, global batch size and
    # data balancing.
    return (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )


def _performance_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the parallelism and memory flags: tensor parallel 4 with
    # sequence parallel, pipeline 1, context 1, expert parallel equal to the whole world size
    # (nodes times GPUs per node), expert tensor 1, full uniform recompute of 1 layer, dynamic batch
    # size with 16384 tokens per GPU, CPU optimizer offload with overlap and the precision-aware
    # optimizer.
    world_size = args.num_nodes * args.num_gpus_per_node
    return (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {world_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )


def _grpo_args() -> str:
    # [FUNCTION DOC] HELPER: Returns fixed GRPO flags: GRPO advantage estimator, KL loss with
    # coefficient 0.01 and low-variance type, zero entropy coefficient, clip range 0.2 / 0.28.
    return (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )


def _optimizer_args() -> str:
    # [FUNCTION DOC] HELPER: Returns fixed optimizer flags: Adam, learning rate 1e-6 constant,
    # weight decay 0.1, betas 0.9 and 0.98.
    return (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )


def _sglang_args() -> str:
    # [FUNCTION DOC] HELPER: Returns fixed SGLang flags: 1 GPU per engine, static memory fraction
    # 0.7, glm47 tool call parser, glm45 reasoning parser, router port 31000.
    return (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-router-port 31000 "
    )


def _agent_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the agent flags: the agentic tool-call generator, the Harbor
    # example's agent function, reward function and rollout class (resolved through the python path),
    # the no-aborted dynamic sampling filter, glm47 TITO model, session server v2, the bind ip flag
    # only if set, the external host flag only if set, port 30000 and 32 workers.
    bind_arg = f"--session-server-ip {args.session_server_ip} " if args.session_server_ip else ""
    external_host_arg = (
        f"--session-server-external-host {args.session_server_external_host} "
        if args.session_server_external_host
        else ""
    )
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server v2 "
        f"{bind_arg}"
        f"{external_host_arg}"
        "--session-server-port 30000 "
        "--session-server-workers 32 "
    )


def _misc_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns miscellaneous flags: dropouts off, fp32 gradient accumulation
    # and softmax, flash attention, training entropy observation, multi-turn logging, colocated
    # mode, actor nodes and GPUs per node, GPUs per node, and the rollout-only debug flag in debug
    # mode.
    debug_arg = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    return (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--observe-training-entropy "
        "--log-multi-turn "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"{debug_arg}"
    )


def _observability_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns monitoring flags: Prometheus (enable, port, run name equal to
    # the run id) only if enabled; details dump to the trace dir; the dashboard; rollout entropy;
    # and the default W&B flags (active only when a key is set in the environment) with the
    # run name prefix terminus-compaction and the run id.
    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = (
            "--use-prometheus " f"--prometheus-port {args.prometheus_port} " f"--prometheus-run-name {args.run_id} "
        )
    return (
        f"--dump-details {_save_traces_dir(args)} "
        "--use-miles-dashboard "
        "--use-rollout-entropy "
        f"{prometheus_args}"
        f"{command_utils.get_default_wandb_args(__file__, run_name_prefix='terminus-compaction', run_id=args.run_id)}"
    )


def _build_train_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Concatenates the flag blocks in the order checkpoint, rollout,
    # optimizer, GRPO, observability, performance, SGLang, agent, misc (no separators; each block
    # ends with a space) and returns the full argument string.
    return "".join(
        (
            _checkpoint_args(args),
            _rollout_args(args),
            _optimizer_args(),
            _grpo_args(),
            _observability_args(args),
            _performance_args(args),
            _sglang_args(),
            _agent_args(args),
            _misc_args(args),
        )
    )


def _extra_env_vars(args: ScriptArgs) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns the job runtime environment mapping: python path set to the
    # Harbor example directory, agent server url, agent model name, and trial timeout (as text);
    # plus the host ip variable only if configured.
    env = {
        "PYTHONPATH": str(_HARBOR_PIPELINE_DIR),
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "AGENT_TRIAL_TIMEOUT": str(args.agent_trial_timeout),
    }
    if args.miles_host_ip:
        env["MILES_HOST_IP"] = args.miles_host_ip
    return env


def _prepare(args: ScriptArgs) -> None:
    # [FUNCTION DOC] HELPER: Converts the HF checkpoint to the Megatron distributed format under
    # the model dir through the backend's convert-checkpoint command. Side effects: runs commands,
    # uses GPUs.
    U = args.create_backend()
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=_hf_checkpoint(args),
        megatron_path=args.megatron_path,
    )


def _execute(args: ScriptArgs) -> None:
    # [FUNCTION DOC] HELPER: Submits the training job through the backend's execute-train with the
    # assembled argument string, GPUs per node, model type, megatron path and the runtime
    # environment mapping. Side effects: starts Ray and the job.
    U = args.create_backend()
    U.execute_train(
        train_args=_build_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=_extra_env_vars(args),
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Entry point (command line generated from the config dataclass): converts the
    # checkpoint unless skip_prepare is set, then submits the job.
    if not args.skip_prepare:
        _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
