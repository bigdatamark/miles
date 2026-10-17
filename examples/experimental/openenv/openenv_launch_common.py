# [MODULE DOC] HELPER module: model-agnostic building blocks for the OpenEnv tbench2 launchers.
# Each function returns a fragment of the training argument string (rollout, GRPO, optimizer,
# agent wiring, W&B, Prometheus) or environment-variable mapping for the ray runtime env, plus a
# process cleanup routine and the sandbox backend resolver. A protocol class documents which
# configuration fields the helpers read from a launcher's config object.
"""Shared launch helpers for the OpenEnv tbench2 learning launchers.

``run-openenv-tbench2.py`` (GLM-4.7-Flash) is the launcher in this example;
sibling per-model launchers (e.g. a DeepSeek-V4-Flash variant) reuse the same
agentic adapter and differ only in the model-family serving/training profile.
The model-agnostic fragments (process cleanup, GRPO/optimizer/rollout/agent
flags, W&B + Prometheus wiring, and the OpenEnv env-var plumbing) live here so
those launchers cannot silently drift apart. Each launcher keeps only its own
perf/sglang/misc profile and its ``ScriptArgs`` defaults.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Protocol

import openenv_sandbox_common as sandbox_common
from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, provision_provider


class LaunchArgs(Protocol):
    # [CLASS DOC] HELPER: Structural type listing the config attributes the helpers read: data and
    # batch sizes (prompt data, rollout batch, samples per prompt, max sequence length, global
    # batch), OpenEnv settings (env url, agent model name, turn cap, rollout time cap, task dir,
    # sandbox backend, credential-file paths for Daytona, E2B and Modal, session server external
    # host, host ip), W&B settings (key, project, team, run name) and Prometheus settings (enable
    # flag, port, run name). No behavior.
    """The config fields the shared helpers read (satisfied by each launcher's ScriptArgs)."""

    prompt_data: str
    rollout_batch_size: int
    n_samples_per_prompt: int
    max_seq_len: int
    global_batch_size: int

    openenv_env_url: str
    agent_model_name: str
    openenv_max_turns: int
    openenv_max_rollout_time_seconds: int
    openenv_tb2_tasks_dir: str
    openenv_sandbox_backend: str
    daytona_api_key_file: str
    e2b_api_key_file: str
    modal_config_file: str
    session_server_external_host: str
    miles_host_ip: str

    wandb_key: str
    wandb_project: str
    wandb_team: str
    wandb_run_name: str

    use_prometheus: bool
    prometheus_port: int
    prometheus_run_name: str


def cleanup() -> None:
    # [FUNCTION DOC] HELPER: Prints start and end messages and, for each of the patterns sglang,
    # train.py and MegatronTrain, runs a shell pipeline that lists matching process ids, excludes
    # this process and its parent, and kills the rest (errors ignored). Sleeps 5 seconds at the
    # end. Side effects: terminates other processes on the host.
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def rollout_args(args: LaunchArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the rollout flags: prompt data from the config with input key
    # prompt and metadata key metadata, shuffle, 40 rollouts, rollout batch and samples per prompt
    # and max sequence length and global batch size from the config, temperature 0.8, response
    # length limit 8192, data balancing. The rollout count, temperature and response limit are
    # fixed, not configurable.
    return (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        "--num-rollout 40 "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        "--rollout-max-response-len 8192 "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )


def grpo_args() -> str:
    # [FUNCTION DOC] HELPER: Returns fixed GRPO flags: GRPO advantage estimator, KL loss with
    # coefficient 0.01 and low-variance type, zero entropy coefficient, clip range 0.2 low and 0.28
    # high.
    return (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )


def optimizer_args() -> str:
    # [FUNCTION DOC] HELPER: Returns fixed optimizer flags: Adam, learning rate 1e-6 with constant
    # schedule, weight decay 0.1, betas 0.9 and 0.98.
    return (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )


def resolve_sandbox_backend(args: LaunchArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the canonical sandbox backend name, or an empty string when
    # the shared env server mode is in effect. Reads the backend name (trimmed; missing attribute
    # treated as empty) and the task directory from the config. Both empty -> empty result. Task
    # directory missing but backend named -> value error; backend missing but task directory given
    # -> value error; otherwise the name is normalized and validated by the common registry (which
    # raises a value error for unknown names).
    """The per-episode sandbox backend in effect, or "" for the shared env server.

    Names and aliases resolve through openenv_sandbox_common, the canonical
    registry, so the accepted set is never enumerated twice: "agentenv" is an
    accepted alias for "e2b", because AgentENV
    (https://github.com/kvcache-ai/AgentENV) is a self-hosted Firecracker
    microVM platform whose native API is the E2B API, so it runs on the e2b
    backend with E2B_API_URL/E2B_SANDBOX_URL pointed at it.

    The two settings that turn this mode on come as a pair — a task checkout
    to build images from, and the provider to build them on — so naming one
    without the other is an error rather than a guess.
    """
    raw = (getattr(args, "openenv_sandbox_backend", "") or "").strip()
    tasks_dir = args.openenv_tb2_tasks_dir
    if not raw and not tasks_dir:
        return ""
    if not tasks_dir:
        raise ValueError(
            "sandbox backends build per-task images: set openenv_tb2_tasks_dir to a terminal-bench-2 checkout"
        )
    if not raw:
        raise ValueError(
            "openenv_tb2_tasks_dir selects per-episode sandboxes: set openenv_sandbox_backend "
            "(OPENENV_SANDBOX_BACKEND) to the provider to run them on"
        )
    return sandbox_common.resolve_backend(raw)


def agent_args(tito_model: str, sandbox_backend: str = "", session_server_external_host: str = "") -> str:
    # [FUNCTION DOC] HELPER: Returns the agentic rollout flags: the agentic tool-call generator, the
    # agent function path (the backend's run function from the registry, falling back to the shared
    # server run function for an empty or unknown backend), the OpenEnv reward hook, the no-aborted
    # dynamic sampling filter, the given TITO model, session server enabled on port 30000 with 32
    # workers, and the external host flag only if a host is given.
    """Agentic-rollout wiring. The TITO surface differs across models; the
    agent function decides where episodes run — per-episode sandboxes on
    whichever backend the launcher resolves (see resolve_sandbox_backend), else
    the one shared env server."""
    agent_fn = sandbox_common.AGENT_FUNCTIONS.get(sandbox_backend, "openenv_agent_function.run")
    external_host_arg = (
        f"--session-server-external-host {session_server_external_host} " if session_server_external_host else ""
    )
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        f"--custom-agent-function-path {agent_fn} "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--tito-model {tito_model} "
        "--use-session-server "
        "--session-server-port 30000 "
        "--session-server-workers 32 "
        f"{external_host_arg}"
    )


def wandb_args(args: LaunchArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns W&B flags, or an empty string when no key is configured.
    # Otherwise enables W&B with the configured project, the run name as group, the key (placed in
    # the command line), and the team only if set.
    if not args.wandb_key:
        return ""
    out = (
        "--use-wandb "
        f"--wandb-project {args.wandb_project} "
        f"--wandb-group {args.wandb_run_name} "
        f"--wandb-key {args.wandb_key} "
    )
    if args.wandb_team:
        out += f"--wandb-team {args.wandb_team} "
    return out


def prometheus_args(args: LaunchArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns Prometheus flags (enable, port, run name) or an empty string
    # when disabled in the config.
    if not args.use_prometheus:
        return ""
    return (
        "--use-prometheus "
        f"--prometheus-port {args.prometheus_port} "
        f"--prometheus-run-name {args.prometheus_run_name} "
    )


def base_env_vars(args: LaunchArgs, script_dir: str, megatron_path: str, miles_root: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns the base runtime environment mapping: python path joining
    # megatron path, script dir and miles root with colons (in that order), plus the env server url,
    # turn cap and rollout time cap (as text) and agent model name from the config.
    return {
        "PYTHONPATH": f"{megatron_path}:{script_dir}:{miles_root}",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(args.openenv_max_rollout_time_seconds),
        "AGENT_MODEL_NAME": args.agent_model_name,
    }


def apply_optional_env_vars(env: dict[str, str], args: LaunchArgs) -> None:
    # [FUNCTION DOC] HELPER: Mutates the given environment mapping in place; returns nothing.
    # Sets the host ip variable when configured. Resolves the sandbox backend; if one is in effect:
    # looks up its credential spec in the provider table, provisions the provider credentials into
    # the mapping (passing the configured credential-file path for that provider, or empty), then
    # preflights the env package: imports it (a missing package raises a runtime error with install
    # advice), reads the server environment source file next to it, and raises a runtime error
    # unless that source mentions both the test-withholding setting and the canonical-verdict
    # requirement (an older install would silently mis-score). Finally sets the task directory
    # variable from the config. Side effects: imports a package and reads a source file.
    """Add the node-address and sandbox-provider env vars the args ask for."""
    if args.miles_host_ip:
        env["MILES_HOST_IP"] = args.miles_host_ip
    backend = resolve_sandbox_backend(args)
    if backend:
        spec = PROVIDER_CREDENTIALS[backend]
        provision_provider(env, spec, arg_path=getattr(args, spec["arg_attr"], "") or "")
        # Preflight the env package the recipe bakes into each task image —
        # shared by every sandbox backend. The import check catches a missing
        # install; the source probe catches an install that imports fine but
        # lacks the server features the sandbox backends score through (canonical
        # tests/test.sh evaluate, TB2_WITHHOLD_TESTS) — that one would not
        # even fail per-episode, it would silently mis-score every episode.
        try:
            import tbench2_env
        except ImportError as e:
            raise RuntimeError(
                "the sandbox modes need tbench2_env in the rollout "
                "process's environment: pip install -e '<OpenEnv>/envs/tbench2_env' "
                "from the checkout described in this directory's README"
            ) from e
        server_src = Path(tbench2_env.__file__).resolve().parent / "server" / "tbench2_env_environment.py"
        src_text = server_src.read_text(encoding="utf-8") if server_src.is_file() else ""
        # `_require_canonical_verdict` (#1025) is what turns a verifier that never
        # wrote reward.txt into an error; before it, that reply was reward 0.0
        # WITH the harness marker, which the per-episode guard cannot tell from
        # a genuine failure.
        if "TB2_WITHHOLD_TESTS" not in src_text or "_require_canonical_verdict" not in src_text:
            raise RuntimeError(
                "the installed tbench2_env server lacks the native-evaluate "
                "contract (canonical test.sh scoring / TB2_WITHHOLD_TESTS / "
                "missing verdict reported as an error): install from an OpenEnv "
                "checkout at or after the #1025 merge (38b2a3135) — see this "
                "directory's README"
            )
        env["OPENENV_TB2_TASKS_DIR"] = args.openenv_tb2_tasks_dir
