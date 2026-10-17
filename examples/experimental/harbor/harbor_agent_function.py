# [MODULE DOC]
# Agent function plug-in (loaded through the custom agent function path option of the agentic tool
# call generator) that runs one Harbor trial inside the rollout worker process for each sample.
# Flow of the async entry point: resolve the provider API key file into the environment, build a
# Harbor trial configuration from the sample metadata, environment variables and the session URL
# of the training-side inference session server, create and run the trial under a wall-clock
# timeout, then convert Harbor's trial result into a metadata mapping with reward, exit status,
# verifier report and agent metrics. Per-harness settings (which keyword arguments and
# environment variables carry the endpoint, key and model name) live in a table of bindings
# keyed by agent name, with a default for any other OpenAI-chat-completions-speaking agent.
# Configuration errors raise (they would fail every sample); trial-level failures map to a
# zero reward with a named exit status. Harbor itself is imported lazily inside functions so the
# module imports without it. All settings are read from environment variables (listed in the
# existing documentation below), never from arguments.
"""In-process Harbor agent function for ``agentic_tool_call.generate``.

Runs one Harbor trial inside the rollout worker: build a ``TrialConfig`` from
the sample, ``Trial.run()`` it, and hand the verdict back as sample metadata.
Nothing sits between the worker and Harbor -- no agent server -- so this needs a
sandbox backend the worker can drive over the network (``HARBOR_ENV_TYPE`` =
``e2b`` for E2B Cloud or a self-hosted E2B-compatible endpoint, ``daytona``, ``modal``, ...).
The local ``docker`` backend needs a Docker daemon next to ``Trial.run()``;
for that, keep the agent server (``examples/swe-agent-harbor-docker``).

Which Harbor: the ``harbor-miles-*`` branch of harbor-framework/harbor (see
the README for the install line). Of what that branch adds on top of upstream,
this path uses:

  needed        terminus-2 ``response_length_exceeded_policy`` (upstream
                salvages a truncated reply as a full assistant turn, which the
                TITO session server rejects); ``MaxSeqLenExceededError`` /
                ``SingleTurnMaxSeqLenExceededError`` (mapped to
                ``SequenceLengthLimitExceeded`` below); the ``max_seq_len``
                check in ``Chat`` and mini-swe-agent's ``poll_steps``.
  useful        mini-swe-agent's output/trajectory caps and per-step timing
                (``agent_metrics``); claude-code's ``n_steps``.
  not used      ``agent_server/``, the dashboard, the docker egress-control
                and network-prune patches, the daytona create jitter, the
                critic agents, S3 upload, adapters.

Env vars (read on the rollout worker):
  HARBOR_TASKS_DIR       directory holding one Harbor task dir per
                         ``metadata.instance_id`` (required)
  HARBOR_ENV_TYPE        Harbor ``EnvironmentType`` value (required; no default
                         -- it decides whose quota a run spends)
  HARBOR_ENV_KWARGS      JSON object passed as ``EnvironmentConfig.kwargs``
                         (backend-specific, e.g. Daytona's auto_snapshot)
  E2B_API_KEY_FILE / DAYTONA_API_KEY_FILE / ...
                         the launcher forwards the provider key by file PATH;
                         the worker resolves it into the SDK's env var here
                         (raises if neither the var nor a usable file exists)
  HARBOR_TRIALS_DIR      where Harbor writes trial dirs (default /tmp/harbor_trials)
  AGENT_TIMEOUT          Harbor's per-trial agent timeout in seconds; the
                         wall-clock cap AGENT_TRIAL_TIMEOUT (default 7200) sits
                         above it and scores the trial 0 if Harbor never returns
  AGENT_MODEL_NAME       model name handed to the agent (default "model")
  AGENT_MAX_INPUT_TOKENS / AGENT_MAX_OUTPUT_TOKENS, HARBOR_MAX_SEQ_LEN,
  HARBOR_AGENT_MAX_ITERATIONS, HARBOR_RESPONSE_LENGTH_POLICY,
  HARBOR_TERMINUS_2_ENABLE_SUMMARIZE, HARBOR_TERMINUS_2_LINEAR_HISTORY,
  HARBOR_OVERRIDE_CPUS, HARBOR_OVERRIDE_MEMORY_MB,
  HARBOR_CPU_ENFORCEMENT_POLICY, HARBOR_MEMORY_ENFORCEMENT_POLICY,
  HARBOR_TIMEOUT_MULTIPLIER,
  HARBOR_VERIFIER_TIMEOUT_SEC, HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER,
  HARBOR_AGENT_ALLOWED_HOSTS
                         same meaning as on the agent server
  HARBOR_OVERRIDE_STORAGE_MB
                         per-sandbox disk, Harbor's ``override_storage_mb``

Failure semantics: a verdict is returned as-is; every episode that ends
without one scores 0 with a named ``exit_status`` (``TimeLimitExceeded``,
``SequenceLengthLimitExceeded``, ``AgentError``), matching the agent-server
path. Configuration errors (missing task dir, bad env vars) raise instead:
they would fail every sample, and a loud stop beats training on silent
all-zero rewards. Nothing is discarded here yet; see the tracking issue for
wiring the platform-side Harbor exceptions to ``InfraAbort`` once that
contract lands.
"""

import asyncio
import json
import logging
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, resolve_provider_api_key
from miles.rollout.agentic.session import openai_session_url

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_TRIAL_TIMEOUT_S = 7200
_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Harbor exception class names -> the exit_status vocabulary the reward path reads.
_TIMEOUT_EXCEPTIONS = {"AgentTimeoutError", "VerifierTimeoutError", "EnvironmentStartTimeoutError"}
_OUTPUT_LIMIT_EXCEPTIONS = {"MaxSeqLenExceededError", "SingleTurnMaxSeqLenExceededError"}


def _trial_timeout_s() -> int:
    # [FUNCTION DOC] HELPER: Returns the wall-clock cap in whole seconds for one trial, read from
    # the trial timeout environment variable (default 7200). A non-integer value raises a value
    # error.
    """The wall-clock cap run() puts on one trial; the longest a trial can live."""
    return int(os.environ.get("AGENT_TRIAL_TIMEOUT", _DEFAULT_AGENT_TRIAL_TIMEOUT_S))


def _env_flag(var: str) -> bool:
    # [FUNCTION DOC] HELPER: Reads a boolean environment variable: true when its value
    # (default "false"), lowercased, is one of 1, true or t; every other value, including yes, is
    # false.
    return os.getenv(var, "false").lower() in ("1", "true", "t")


def _env_int(var: str) -> int | None:
    # [FUNCTION DOC] HELPER: Reads an integer environment variable; returns none when unset or
    # empty, otherwise the parsed integer (a non-integer value raises a value error).
    raw = os.getenv(var)
    return int(raw) if raw else None


def _allowed_hosts(var: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Splits the environment variable on commas and whitespace and returns
    # the non-empty host names in order (empty list if unset).
    return [host for host in re.split(r"[,\s]+", os.getenv(var, "")) if host]


# --- harness bindings -------------------------------------------------------
#
# What Miles must know per harness to hand it the session URL: which env vars /
# kwargs carry the endpoint and key, and what the model is called on its wire.
# Everything else about running the harness is Harbor's.
#
# Ported from harbor-framework/harbor agent_server/trial_runner.py
# ``_agent_connection_config`` at harbor-miles-v0.20.0 (53a6e92a). That copy
# keeps serving the agent-server path and keeps evolving; when bumping the
# harbor pin, diff that function against this table. The tests assert WHY each
# line exists (terminus-2 must abort on truncation, opencode must not resolve
# the ``openai`` provider id, ...), so a re-sync knows which lines carry a
# constraint and which just follow the harness.


@dataclass(frozen=True)
class HarnessBinding:
    # [CLASS DOC]
    # Immutable per-harness adapter with three members: a function building the agent keyword
    # arguments from the session URL, API key, sampling parameters mapping, model name and optional
    # maximum sequence length; a function building the agent environment variable mapping from the
    # session URL and API key; and a function mapping the model name to the name Harbor passes to the
    # harness (identity by default).
    # (session_url, api_key, sampling_params, model, max_seq_len) -> AgentConfig.kwargs
    kwargs: Callable[[str, str, dict[str, Any], str, int | None], dict[str, Any]]
    # (session_url, api_key) -> AgentConfig.env
    env: Callable[[str, str], dict[str, str]]
    # the model name Harbor passes to the harness
    model_name: Callable[[str], str] = lambda model: model


def _terminus_kwargs(
    session_url: str, api_key: str, sampling_params: dict[str, Any], model: str, max_seq_len: int | None
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Agent keyword arguments for the terminus family. Fixed: XML parser,
    # interleaved thinking, abort when a response exceeds the length limit, the sampling parameters
    # as LLM call keyword arguments, the session URL as API base, the API key inside the LLM keyword
    # arguments (the agent has no top-level key parameter), summarisation and linear history flags
    # from environment booleans, and the response-length policy from its environment variable
    # (default abort, because recovering would replay a fabricated assistant turn and desync the
    # token-level session server). If the maximum-iterations environment integer is set and non-zero
    # it adds a turn cap and suppresses the cap warning. The model and maximum sequence length
    # parameters are unused here.
    kwargs: dict[str, Any] = {
        "parser_name": "xml",
        "interleaved_thinking": True,
        "abort_on_response_length_exceeded": True,
        "llm_call_kwargs": dict(sampling_params),
        "api_base": session_url,
        # Terminus2 has no top-level api_key parameter: the key must ride
        # llm_kwargs (the LLM constructor extras, spread into every litellm
        # call) or litellm falls back to OPENAI_API_KEY in the process env.
        "llm_kwargs": {"api_key": api_key},
        "enable_summarize": _env_flag("HARBOR_TERMINUS_2_ENABLE_SUMMARIZE"),
        "trajectory_config": {"linear_history": _env_flag("HARBOR_TERMINUS_2_LINEAR_HISTORY")},
        # "abort" ends the trajectory when one reply is truncated at max_tokens;
        # upstream's "recover" re-sends the truncated reply as a fabricated
        # assistant turn, which desyncs the TITO session server.
        "response_length_exceeded_policy": os.getenv("HARBOR_RESPONSE_LENGTH_POLICY", "abort"),
    }
    if max_turns := _env_int("HARBOR_AGENT_MAX_ITERATIONS"):
        kwargs["max_turns"] = max_turns
        kwargs["suppress_max_turns_warning"] = True
    return kwargs


def _openai_env(session_url: str, api_key: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Environment for harnesses speaking OpenAI chat completions: the API
    # key and API base variables set to the key and session URL.
    return {"OPENAI_API_KEY": api_key, "OPENAI_API_BASE": session_url}


def _claude_code_kwargs(
    session_url: str, api_key: str, sampling_params: dict[str, Any], model: str, max_seq_len: int | None
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Agent keyword arguments for the Claude Code harness: disables the web
    # search and web fetch tools (Anthropic server-side tools the session server cannot translate)
    # and, if the maximum-iterations environment integer is set and non-zero, adds a turn cap.
    # All other parameters are unused.
    # WebSearch / WebFetch are Anthropic server-side tools; the session server
    # translates client-side tools onto an OpenAI-compatible backend only.
    kwargs: dict[str, Any] = {"disallowed_tools": "WebSearch,WebFetch"}
    if max_turns := _env_int("HARBOR_AGENT_MAX_ITERATIONS"):
        kwargs["max_turns"] = max_turns
    return kwargs


def _claude_code_env(session_url: str, api_key: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Environment for Claude Code: Anthropic API key set to the key;
    # Anthropic base URL set to the session URL with a trailing version path segment removed (the
    # SDK appends its own messages path and the session server's Anthropic route hangs off the
    # session root); server-side tool search disabled; and, if the max output tokens environment
    # variable is non-empty, the Claude Code output token limit set to the same value.
    env = {
        "ANTHROPIC_API_KEY": api_key,
        # The Anthropic SDK appends /v1/messages itself, and the session
        # server's Anthropic route hangs off the session root -- so hand it
        # the root, not the /v1-suffixed URL the OpenAI-style clients get.
        "ANTHROPIC_BASE_URL": session_url.removesuffix("/v1"),
        # Claude Code's server-side Tool Search cannot be forwarded to the backend.
        "ENABLE_TOOL_SEARCH": "false",
    }
    if max_output_tokens := os.getenv("AGENT_MAX_OUTPUT_TOKENS"):
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = max_output_tokens
    return env


def _mini_swe_agent_env(session_url: str, api_key: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: The OpenAI-style environment plus a setting that makes the mini
    # software-engineering agent ignore cost-tracking errors.
    return {**_openai_env(session_url, api_key), "MSWEA_COST_TRACKING": "ignore_errors"}


def _no_kwargs(
    session_url: str, api_key: str, sampling_params: dict[str, Any], model: str, max_seq_len: int | None
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Keyword-argument builder for harnesses that need none; ignores all
    # parameters and returns an empty mapping.
    return {}


# OpenCode resolves the provider id "openai" through @ai-sdk/openai, which
# issues Responses API calls the session server's chat-completions backend
# rejects. Renaming just the provider id keeps the model id after the slash
# intact, so the request body still names the served model.
_OPENCODE_COMPAT_PROVIDER = "openai-compatible"


def _opencode_provider_model(model: str) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Splits a model name of the form provider slash model id into a pair.
    # A name without a slash gets the compatible provider id and the whole name as model id; a
    # provider equal to openai is renamed to the compatible provider id (the real openai id would
    # make the harness use the Responses API, which the backend rejects), keeping the model id;
    # other providers are kept. Splits at the first slash only.
    provider, sep, model_id = model.partition("/")
    if not sep:
        return _OPENCODE_COMPAT_PROVIDER, model
    if provider == "openai":
        return _OPENCODE_COMPAT_PROVIDER, model_id
    return provider, model_id


def _opencode_model_entry(sampling_params: dict[str, Any], max_seq_len: int | None) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds the OpenCode model entry. A limit mapping gets a context key
    # when a maximum sequence length is given and an output key from the sampling parameters' max
    # tokens, falling back to the max output tokens environment integer, when either is non-zero.
    # Returns a mapping with only the limit key if the limit is non-empty, else an empty mapping.
    """The OpenCode model entry, including its context/output limits.

    OpenCode only auto-compacts a session when it knows the model's context
    window; a model served from a custom provider resolves to 0 and never
    compacts, so a long trial degrades silently. Either key is omitted when
    unknown, so no limit is asserted that cannot be substantiated.
    """
    limit: dict[str, int] = {}
    if max_seq_len:
        limit["context"] = int(max_seq_len)
    if output := (sampling_params.get("max_tokens") or _env_int("AGENT_MAX_OUTPUT_TOKENS")):
        limit["output"] = int(output)
    return {"limit": limit} if limit else {}


def _opencode_kwargs(
    session_url: str, api_key: str, sampling_params: dict[str, Any], model: str, max_seq_len: int | None
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Agent keyword arguments for OpenCode: a configuration mapping
    # defining one provider (id from the model name) using the OpenAI-compatible package, with the
    # session URL as base URL, the API key, and the model entry with limits. No turn cap is set
    # because the agent has no such flag.
    provider, model_id = _opencode_provider_model(model)
    # Deliberately no max_turns: the OpenCode agent exposes no turn-cap flag,
    # so honouring HARBOR_AGENT_MAX_ITERATIONS would be a silent no-op.
    return {
        "opencode_config": {
            "provider": {
                provider: {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": session_url, "apiKey": api_key},
                    "models": {model_id: _opencode_model_entry(sampling_params, max_seq_len)},
                }
            }
        }
    }


def _opencode_env(session_url: str, api_key: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Environment for OpenCode: both the base URL and API base variables set
    # to the session URL (the harness reads the former) and the API key variable.
    # OpenCode reads OPENAI_BASE_URL when deciding whether to write
    # provider.options.baseURL; the generic OPENAI_API_BASE is not consulted.
    # Export both so either lookup resolves.
    return {"OPENAI_BASE_URL": session_url, "OPENAI_API_BASE": session_url, "OPENAI_API_KEY": api_key}


HARNESS_BINDINGS: dict[str, HarnessBinding] = {
    "terminus-2": HarnessBinding(kwargs=_terminus_kwargs, env=_openai_env),
    "terminus-1": HarnessBinding(kwargs=_terminus_kwargs, env=_openai_env),
    "terminus": HarnessBinding(kwargs=_terminus_kwargs, env=_openai_env),
    "claude-code": HarnessBinding(kwargs=_claude_code_kwargs, env=_claude_code_env),
    "opencode": HarnessBinding(
        kwargs=_opencode_kwargs,
        env=_opencode_env,
        model_name=lambda model: "/".join(_opencode_provider_model(model)),
    ),
    "mini-swe-agent": HarnessBinding(kwargs=_no_kwargs, env=_mini_swe_agent_env),
}
# Any other installed agent that speaks OpenAI chat completions.
_DEFAULT_BINDING = HarnessBinding(kwargs=_no_kwargs, env=_openai_env)


# --- trial config ----------------------------------------------------------


def _task_path(tasks_dir: Path, instance_id: str) -> Path:
    # [FUNCTION DOC] HELPER: Resolves the task directory for an instance id under the tasks
    # directory. The id must be non-empty and match a conservative pattern (alphanumeric first
    # character, then alphanumerics, dots, underscores, hyphens) else a value error; the resolved
    # path must be inside the tasks directory (else a value error, guarding symbolic link escapes)
    # and must be an existing directory (else a file-not-found error). Returns the resolved path.
    if not instance_id or not _SAFE_INSTANCE_ID.match(instance_id):
        raise ValueError(f"invalid instance_id {instance_id!r}")
    path = (tasks_dir / instance_id).resolve()
    if tasks_dir.resolve() not in path.parents:
        raise ValueError(f"instance_id {instance_id!r} escapes HARBOR_TASKS_DIR")
    if not path.is_dir():
        raise FileNotFoundError(f"no Harbor task dir for {instance_id!r} under {tasks_dir}")
    return path


def _ensure_provider_key() -> None:
    # [FUNCTION DOC] HELPER: Looks up the credential spec for the (trimmed, lowercased) sandbox
    # backend type; does nothing if the backend has no spec or has more than one key variable. If
    # the single key variable is unset or blank in the environment, resolves the key (from the
    # variable, the key file named by the spec's file variable, or its default path; raising if none
    # works) and stores it in the process environment. Side effect: mutates the environment.
    """Resolve the key file the launcher forwarded (by PATH) into the env var the
    provider's SDK reads. Modal needs nothing here: its SDK reads the config file
    MODAL_CONFIG_PATH names."""
    spec = PROVIDER_CREDENTIALS.get(os.getenv("HARBOR_ENV_TYPE", "").strip().lower())
    if not spec or len(spec["key_env_vars"]) != 1:
        return
    key_env = spec["key_env_vars"][0]
    if not os.getenv(key_env, "").strip():
        os.environ[key_env] = resolve_provider_api_key(key_env, spec["file_env_var"], spec["default_path"])


def _environment_config():
    # [FUNCTION DOC] HELPER: Builds Harbor's environment configuration object (imports Harbor
    # lazily). The backend type comes from the environment variable (trimmed, lowercased); empty
    # raises a value error, an unknown one raises through Harbor's enumeration. Backend keyword
    # arguments come from a JSON object in another variable (default empty object). For the Daytona
    # backend, defaults are filled in: auto-stop interval equals trial timeout in whole minutes
    # (rounded up) plus thirty, auto-delete interval 1440 minutes; an explicit auto-stop interval
    # not exceeding the trial minutes raises a value error. CPU, memory and storage overrides come
    # from integer environment variables (non-positive values raise a value error; unset gives
    # none); CPU and memory enforcement policies come from their variables when non-empty. The
    # configuration always requests sandbox deletion when done.
    """``HARBOR_ENV_TYPE`` straight through to Harbor's ``EnvironmentType``: adding a
    backend is Harbor's job, not a branch here. Backend-specific settings ride in
    ``HARBOR_ENV_KWARGS`` (a JSON object) as ``EnvironmentConfig.kwargs``."""
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import EnvironmentConfig

    raw = os.getenv("HARBOR_ENV_TYPE", "").strip().lower()
    if not raw:
        raise ValueError("set HARBOR_ENV_TYPE to the Harbor environment type to run trials on (e.g. e2b, daytona)")
    env_type = EnvironmentType(raw)  # raises on an unknown backend instead of guessing
    kwargs = json.loads(os.getenv("HARBOR_ENV_KWARGS", "{}") or "{}")
    if env_type == EnvironmentType.DAYTONA:
        # a killed worker never reaches Harbor's teardown, and Harbor's Daytona defaults (0) never reclaim;
        # the stop timer outlasts the trial cap because an in-sandbox agent looks idle to Daytona
        trial_min = math.ceil(_trial_timeout_s() / 60)
        kwargs.setdefault("auto_stop_interval_mins", trial_min + 30)
        kwargs.setdefault("auto_delete_interval_mins", 1440)
        if int(kwargs["auto_stop_interval_mins"]) <= trial_min:
            raise ValueError(
                f"auto_stop_interval_mins={kwargs['auto_stop_interval_mins']} would stop a Daytona sandbox "
                f"mid-trial: AGENT_TRIAL_TIMEOUT allows {trial_min} minutes"
            )
    overrides = {}
    for field, var in (
        ("override_cpus", "HARBOR_OVERRIDE_CPUS"),
        ("override_memory_mb", "HARBOR_OVERRIDE_MEMORY_MB"),
        ("override_storage_mb", "HARBOR_OVERRIDE_STORAGE_MB"),
    ):
        value = _env_int(var)
        if value is not None and value <= 0:
            raise ValueError(f"{var} must be a positive integer")
        overrides[field] = value
    policies = {}
    for field, var in (
        ("cpu_enforcement_policy", "HARBOR_CPU_ENFORCEMENT_POLICY"),
        ("memory_enforcement_policy", "HARBOR_MEMORY_ENFORCEMENT_POLICY"),
    ):
        if value := os.getenv(var, "").strip():
            policies[field] = value
    return EnvironmentConfig(type=env_type, delete=True, kwargs=kwargs, **overrides, **policies)


def build_trial_config(metadata: dict[str, Any], session_url: str, request_kwargs: dict[str, Any]):
    # [FUNCTION DOC] Builds Harbor's trial configuration (imports Harbor lazily). Inputs: the
    # sample metadata mapping (agent name, default mini software-engineering agent; instance id;
    # optional maximum sequence length), the session URL and the sampling parameters mapping.
    # The tasks directory comes from its required environment variable (key error if unset). The
    # model name is "openai/" plus the model name environment value (default "model") and the API
    # key is a fixed dummy value. The harness binding is looked up by agent name with the default
    # binding as fallback. Maximum sequence length is the metadata value or the environment
    # integer, if any. Agent keyword arguments from the binding are extended with model info
    # (maximum input tokens default 32768, maximum output tokens default 8192, zero costs) and the
    # sequence length when known. Optional verifier timeout and environment build timeout
    # multiplier come from environment floats. The agent config uses the binding's model name,
    # the agent timeout from its environment float (none if unset), the binding environment, the
    # keyword arguments and extra allowed hosts. Also sets the task path (validated), the environment
    # configuration, the trials directory (default under the system temp directory) and the timeout
    # multiplier (default 2.0). Returns the configuration object.
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig, VerifierConfig

    tasks_dir = Path(os.environ["HARBOR_TASKS_DIR"])
    agent_name = str(metadata.get("agent_name") or "mini-swe-agent")
    model = f"openai/{os.getenv('AGENT_MODEL_NAME', 'model')}"
    api_key = "dummy"
    binding = HARNESS_BINDINGS.get(agent_name, _DEFAULT_BINDING)

    raw_max_seq_len = metadata.get("max_seq_len") or _env_int("HARBOR_MAX_SEQ_LEN")
    max_seq_len = int(raw_max_seq_len) if raw_max_seq_len is not None else None
    agent_kwargs = binding.kwargs(session_url, api_key, request_kwargs, model, max_seq_len)
    agent_kwargs["model_info"] = {
        "max_input_tokens": int(os.getenv("AGENT_MAX_INPUT_TOKENS", "32768")),
        "max_output_tokens": int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "8192")),
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    if max_seq_len is not None:
        agent_kwargs["max_seq_len"] = max_seq_len

    extra: dict[str, Any] = {}
    if verifier_timeout := os.getenv("HARBOR_VERIFIER_TIMEOUT_SEC"):
        extra["verifier"] = VerifierConfig(override_timeout_sec=float(verifier_timeout))
    if build_mult := os.getenv("HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER"):
        extra["environment_build_timeout_multiplier"] = float(build_mult)

    agent_timeout = os.getenv("AGENT_TIMEOUT")
    return TrialConfig(
        task=TaskConfig(path=_task_path(tasks_dir, str(metadata.get("instance_id", "")))),
        agent=AgentConfig(
            name=agent_name,
            model_name=binding.model_name(model),
            override_timeout_sec=float(agent_timeout) if agent_timeout else None,
            env=binding.env(session_url, api_key),
            kwargs=agent_kwargs,
            extra_allowed_hosts=_allowed_hosts("HARBOR_AGENT_ALLOWED_HOSTS"),
        ),
        environment=_environment_config(),
        trials_dir=Path(os.getenv("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")),
        timeout_multiplier=float(os.getenv("HARBOR_TIMEOUT_MULTIPLIER", "2.0")),
        **extra,
    )


# --- result mapping --------------------------------------------------------


def _timing_duration_sec(timing) -> float | None:
    # [FUNCTION DOC] HELPER: Returns finish minus start in seconds from an object's start and
    # finish timestamp attributes, or none if either is missing.
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    return (finished - started).total_seconds() if started and finished else None


def trial_result_to_metadata(result) -> dict[str, Any]:
    # [FUNCTION DOC] Converts Harbor's trial result into a mapping with reward, exit status,
    # eval report and agent metrics. Exit status: if the result has exception info, it is time
    # limit exceeded for the agent, verifier or environment-start timeout exception types, sequence
    # length limit exceeded for the two max-sequence-length exception types, otherwise agent error;
    # without exception info it is submitted when a verifier result exists, else agent error.
    # Rewards: the verifier's reward mapping (copied) is the eval report; the reward is its
    # "reward" entry, or its first value if that key is missing, or 0.0 when the mapping is empty
    # or there is no verifier result. Metrics: input tokens, output tokens and cost when not none;
    # the step count under the name turns; any agent metadata mapping merged in (overriding on
    # equal keys); and durations in seconds for total, environment setup, agent setup, agent run
    # and evaluation phases when their timestamps exist. All attribute accesses are tolerant of
    # missing attributes.
    """Harbor ``TrialResult`` -> the dict merged into sample metadata."""
    exc = getattr(result, "exception_info", None)
    if exc is not None:
        exc_type = getattr(exc, "exception_type", "")
        if exc_type in _TIMEOUT_EXCEPTIONS:
            exit_status = "TimeLimitExceeded"
        elif exc_type in _OUTPUT_LIMIT_EXCEPTIONS:
            exit_status = "SequenceLengthLimitExceeded"
        else:
            exit_status = "AgentError"
    elif getattr(result, "verifier_result", None) is not None:
        exit_status = "Submitted"
    else:
        exit_status = "AgentError"

    verifier = getattr(result, "verifier_result", None)
    rewards = dict(getattr(verifier, "rewards", None) or {}) if verifier is not None else {}
    reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0))) if rewards else 0.0

    metrics: dict[str, Any] = {}
    agent_result = getattr(result, "agent_result", None)
    if agent_result is not None:
        for field in ("n_input_tokens", "n_output_tokens", "cost_usd"):
            if (value := getattr(agent_result, field, None)) is not None:
                metrics[field] = value
        if (n_steps := getattr(agent_result, "n_steps", None)) is not None:
            metrics["turns"] = n_steps
        if isinstance(agent_meta := getattr(agent_result, "metadata", None), dict):
            metrics.update(agent_meta)
    for key, timing in {
        "total_time": result,
        "env_setup_time": getattr(result, "environment_setup", None),
        "agent_setup_time": getattr(result, "agent_setup", None),
        "agent_run_time": getattr(result, "agent_execution", None),
        "eval_time": getattr(result, "verifier", None),
    }.items():
        if timing is not None and (duration := _timing_duration_sec(timing)) is not None:
            metrics[key] = duration

    return {"reward": reward, "exit_status": exit_status, "eval_report": rewards, "agent_metrics": metrics}


def _failed(exit_status: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Result mapping for an episode without a verdict: reward 0.0, the
    # given exit status, and empty eval report and agent metrics.
    return {"reward": 0.0, "exit_status": exit_status, "eval_report": {}, "agent_metrics": {}}


# --- entry -----------------------------------------------------------------


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    # [FUNCTION DOC] Async entry point called per sample (extra keyword arguments accepted and
    # ignored). Inputs: the inference base URL (converted to the OpenAI-style session URL), the
    # prompt, sampling request keyword arguments (default empty) and the sample metadata (default
    # empty). Resolves the provider key and builds the trial configuration first, outside any
    # error handling, so configuration errors propagate and stop the run. Then creates the trial
    # and runs it under the wall-clock timeout; a timeout is logged and returns the failed result
    # with exit status time limit exceeded; any other exception is logged with traceback and
    # returns the failed result with agent error. On success returns the converted result mapping
    # plus the trial directory path as text, logging exit status and reward.
    # KNOWN ISSUE: the prompt parameter is unused; the instruction comes from the Harbor task
    # directory selected by the instance id in the metadata.
    """Run one Harbor trial for the sample; return its verdict as metadata."""
    from harbor.trial.trial import Trial

    metadata = metadata or {}
    request_kwargs = request_kwargs or {}
    session_url = openai_session_url(base_url)
    instance_id = metadata.get("instance_id")
    trial_timeout_s = _trial_timeout_s()

    # Config errors (missing task dir, bad env vars, unresolvable provider key)
    # fail every sample the same way; raising beats training on silent all-zero
    # rewards.
    _ensure_provider_key()
    config = build_trial_config(metadata, session_url, request_kwargs)

    try:
        trial = await Trial.create(config)
        # wait_for cancels the coroutine on expiry; Trial.run handles the
        # cancellation and stops its environment.
        result = await asyncio.wait_for(trial.run(), timeout=trial_timeout_s)
    except asyncio.TimeoutError:
        # The policy may be what is stalling, so this is a negative sample.
        logger.error(f"Harbor trial for {instance_id} exceeded {trial_timeout_s}s; scoring 0")
        return _failed("TimeLimitExceeded")
    except Exception as e:
        logger.error(f"Harbor trial for {instance_id} failed: {e}", exc_info=True)
        return _failed("AgentError")

    out = trial_result_to_metadata(result)
    out["trial_dir"] = str(trial.paths.trial_dir)
    logger.info(f"Harbor trial {instance_id}: exit_status={out['exit_status']} reward={out['reward']}")
    return out
