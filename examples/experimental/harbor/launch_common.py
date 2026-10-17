# [MODULE DOC]
# Shared helpers for the Harbor example launchers. Provides the directories to add to the
# workers' module search path (this example's directory and the sibling agent-server example, whose
# reward hook and rollout class are reused), the fixed block of training command line flags that
# wire up the agentic tool-call generation with the Harbor agent function and the inference session
# server, and a preflighted mapping of Harbor-related environment variables to forward to rollout
# workers. Kept free of heavy training dependencies; it needs only the credential helper module.
"""The Harbor-side launcher wiring, shared by this example's launchers.

Separate from run.py so it imports without the trainer's heavy dependencies:
the launchers pull in miles' command utils, while this module needs only the
credential contract.
"""

import os
from pathlib import Path

from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, provision_provider

HARBOR_EXAMPLE_DIR = Path(__file__).resolve().parent
# reward_func / RolloutFn (agent-metric aggregation) are reused from the agent-server example
HARBOR_DOCKER_EXAMPLE_DIR = HARBOR_EXAMPLE_DIR.parents[1] / "swe-agent-harbor-docker"


def agentic_pythonpath_dirs() -> list[str]:
    # [FUNCTION DOC] Returns the two absolute directory paths (as text) to put on the workers'
    # module search path: this example directory, then the swe-agent-harbor-docker example
    # directory (two levels up from here, under the top-level examples directory). Existence is
    # not checked.
    """Directories every Harbor launcher puts on the workers' PYTHONPATH: the
    agent function here, and the reward hook it reuses from the agent-server
    example."""
    return [str(HARBOR_EXAMPLE_DIR), str(HARBOR_DOCKER_EXAMPLE_DIR)]


def agentic_train_args(
    *,
    tito_model: str,
    session_server_workers: int,
    session_server_port: int = 30000,
    session_server_external_host: str = "",
) -> str:
    # [FUNCTION DOC] Builds the flag string (keyword-only inputs: token-in-token-out model name,
    # session server worker count, session server port default 30000, optional external host
    # default empty). The flags: custom generate function path set to the agentic tool call
    # generator; custom agent function path set to the Harbor agent function's run; custom reward
    # path and rollout function path set to the reward function and rollout class of the reused
    # example's generate module; the dynamic sampling filter that drops aborted samples; the model
    # name for token handling; use of the session server with its port and worker count; and, only
    # when an external host is given, the external host flag. Returns one string with each flag
    # followed by a space.
    """The agentic wiring every Harbor launcher passes to train.py.

    One copy, shared by the recipes and the GPU e2e, so the flags the test
    exercises are the flags the recipes ship. Model-specific values come from
    the caller.
    """
    external_host_arg = (
        f"--session-server-external-host {session_server_external_host} " if session_server_external_host else ""
    )
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path harbor_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--tito-model {tito_model} "
        "--use-session-server "
        f"--session-server-port {session_server_port} "
        f"--session-server-workers {session_server_workers} "
        f"{external_host_arg}"
    )


def harbor_env_vars(args) -> dict[str, str]:
    # [FUNCTION DOC] Validates the launcher arguments and builds the environment mapping for
    # rollout workers. The sandbox backend type is trimmed and lowercased; empty raises a value
    # error asking to set it, and docker raises a value error (docker needs a daemon next to the
    # trial, use the agent-server example). Harbor must be importable in the launching process, else
    # a runtime error. The mapping holds backend type, tasks directory, trials directory, agent
    # model name and agent timeout (as text), plus the backend keyword JSON when non-empty. For a
    # backend with a known credential spec, the credential is provisioned by file path (using the
    # launcher argument named in the spec, empty if absent); for other backends a notice is printed
    # that the worker environment must carry the credentials. Finally forwards, when non-blank in
    # the launcher's environment (trimmed), a fixed list of tuning variables (token limits, trial
    # timeout, sequence length, iteration cap, response length policy, terminus options, resource
    # overrides and policies, timeout multipliers, verifier timeout, build timeout multiplier,
    # allowed hosts).
    """The Harbor-side environment for rollout workers, preflighted on the launcher.

    HARBOR_ENV_TYPE is passed through untouched (Harbor validates it). The
    provider's credential goes by key-file PATH, its endpoint variables by
    value, on the contract every sandbox backend shares.
    """
    # normalized the same way the agent function reads it, so the guards
    # below see the value the worker will act on
    env_type = args.harbor_env_type.strip().lower()
    if not env_type:
        raise ValueError(
            "set --harbor-env-type / HARBOR_ENV_TYPE (e.g. e2b, daytona): in-process trials need a sandbox backend the worker can reach"
        )
    if env_type == "docker":
        raise ValueError(
            "docker needs a Docker daemon next to Trial.run(); use examples/swe-agent-harbor-docker (agent server) for it"
        )
    try:
        import harbor  # noqa: F401  # the fork branch, see README
    except ImportError as e:
        raise RuntimeError(
            "harbor is not importable in the rollout process's environment; see the README install line"
        ) from e

    env = {
        "HARBOR_ENV_TYPE": env_type,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        "HARBOR_TRIALS_DIR": args.harbor_trials_dir,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "AGENT_TIMEOUT": str(args.agent_timeout),
    }
    if args.harbor_env_kwargs:
        env["HARBOR_ENV_KWARGS"] = args.harbor_env_kwargs
    if spec := PROVIDER_CREDENTIALS.get(env_type):
        provision_provider(env, spec, arg_path=getattr(args, spec["arg_attr"], "") or "")
    else:
        # Any other Harbor backend still passes straight through; there is just
        # no credential wiring known here, so the worker environment must carry
        # whatever that provider's SDK reads.
        print(
            f"harbor: no credential wiring known for {env_type!r}; "
            "assuming the worker environment carries the provider's credentials",
            flush=True,
        )
    # per-server knobs, forwarded when set
    for var in (
        "AGENT_MAX_INPUT_TOKENS",
        "AGENT_MAX_OUTPUT_TOKENS",
        "AGENT_TRIAL_TIMEOUT",
        "HARBOR_MAX_SEQ_LEN",
        "HARBOR_AGENT_MAX_ITERATIONS",
        "HARBOR_RESPONSE_LENGTH_POLICY",
        "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE",
        "HARBOR_TERMINUS_2_LINEAR_HISTORY",
        "HARBOR_OVERRIDE_CPUS",
        "HARBOR_OVERRIDE_MEMORY_MB",
        "HARBOR_OVERRIDE_STORAGE_MB",
        "HARBOR_CPU_ENFORCEMENT_POLICY",
        "HARBOR_MEMORY_ENFORCEMENT_POLICY",
        "HARBOR_TIMEOUT_MULTIPLIER",
        "HARBOR_VERIFIER_TIMEOUT_SEC",
        "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER",
        "HARBOR_AGENT_ALLOWED_HOSTS",
    ):
        if value := os.environ.get(var, "").strip():
            env[var] = value
    return env
