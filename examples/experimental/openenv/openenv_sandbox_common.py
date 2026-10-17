# [MODULE DOC] Provider-independent orchestration of per-episode sandboxes for the OpenEnv
# tbench2 agent functions. Contains: the backend registry (names mapped to the provider agent
# modules and their run entry points, with the alias agentenv for e2b) and its lookup/validation
# helpers; a cancellation-safe single sandbox create that reaps orphans; text-based throttle
# detection; throttle knob reading (create concurrency and max retries from environment variables
# with a provider prefix); and the backend dataclass that throttles creates with a per-rollout
# semaphore, retries with jittered exponential backoff, provides an episode env context manager,
# and exposes the direct-drive episode function and the Miles run entry point.
"""Shared per-episode sandbox orchestration for the backend agent functions.

A backend module (``openenv_daytona_agent_function``,
``openenv_e2b_agent_function``, ``openenv_modal_agent_function``)
owns only what is genuinely provider-specific: how ONE sandbox with the env
server comes into being (its ``tb2_sandbox_*`` materialization module), which
errors count as retryable throttling, and its env-var knobs. Everything that
makes per-episode sandboxes safe under a fanned-out rollout is provider-blind
and lives here once:

  ``create_once``          cancel-safe create. ``asyncio.to_thread`` is not
                cancellable: when an episode's wall-clock cap cancels the
                coroutine mid-create, the worker thread keeps running and its
                (close_fn, url) result would be discarded — leaking a sandbox
                until the provider-side TTL backstop reclaims it. The result
                is recorded thread-side and, on cancellation, handed to a
                reaper that closes the orphan promptly once the create
                finishes.
  ``SandboxBackend``       the orchestration itself. ``start_task_sandbox``
                throttles creates per rollout (``rollout_semaphore``, shared by
                the rollout's agent processes) and retries the errors the backend
                classifies as throttling with jittered exponential backoff —
                anything else propagates immediately, and the semaphore is held
                only for the create attempt and released during backoff so other
                episodes keep the pipeline full.
                ``episode_env`` is the async context manager mirroring one
                episode's lifetime: fresh sandbox -> connected env client ->
                close.
"""

import asyncio
import importlib
import logging
import os
import random
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from miles.rollout.agentic.rollout_limits import rollout_semaphore

# A provider's "start one sandbox" hook: (task_id, tasks_dir) -> (close_fn, base_url).
StartFn = Callable[[str, str], tuple[Callable[[], None], str]]

# The canonical per-episode backend registry — the launcher and the sibling
# tools (scan_golden, eval_tbench2_via_api) all resolve through here, so the
# accepted names and the "agentenv is the e2b backend" aliasing cannot drift.
# Adding a provider is one entry plus its two modules (backend + materialization).
AGENT_MODULES = {
    "daytona": "openenv_daytona_agent_function",
    "e2b": "openenv_e2b_agent_function",
    "modal": "openenv_modal_agent_function",
}
AGENT_FUNCTIONS = {backend: f"{module}.run" for backend, module in AGENT_MODULES.items()}
_ALIASES = {"agentenv": "e2b"}


def backend_names() -> str:
    # [FUNCTION DOC] HELPER: Returns the registered backend names plus aliases, sorted
    # alphabetically and joined with a comma and space, for help and error texts.
    """The accepted names, for help text that must not drift from the registry."""
    return ", ".join(sorted([*AGENT_FUNCTIONS, *_ALIASES]))


def resolve_backend(name: str | None) -> str:
    # [FUNCTION DOC] Normalizes a backend name: trims and lowercases it (none is treated as empty),
    # raises a value error listing the allowed names if empty (there is intentionally no default),
    # maps an alias to its target, and raises a value error if the result is not a registered
    # backend (the message shows the original name). Returns the canonical backend name.
    """Normalize a backend name; reject unknown and missing ones alike.

    There is deliberately no default. Which provider runs decides whose quota
    an entire rollout spends and which credentials must be present, so a name
    left out is a question to answer rather than one to guess at.
    """
    backend = (name or "").strip().lower()
    allowed = backend_names()
    if not backend:
        raise ValueError(f"no sandbox backend named; choose one of: {allowed}")
    backend = _ALIASES.get(backend, backend)
    if backend not in AGENT_FUNCTIONS:
        raise ValueError(f"unknown sandbox backend {name!r}; choose one of: {allowed}")
    return backend


def load_backend(name: str | None) -> "SandboxBackend":
    # [FUNCTION DOC] Resolves the name (may raise a value error) and imports the provider agent
    # module registered for it on demand, returning that module's backend object. Side effect:
    # imports the provider module and so its SDK-dependent code.
    """The SandboxBackend named by *name*, imported on demand.

    The sibling tools (scan_golden, eval_tbench2_via_api) drive one backend
    directly instead of through the training entry point. Resolving it here
    means they gain a provider by naming it, not by growing an import branch
    each — and a provider's SDK is imported only when it is the one selected.
    """
    return importlib.import_module(AGENT_MODULES[resolve_backend(name)]).BACKEND


async def create_once(start_fn: StartFn, task_id: str, tasks_dir: str, *, logger: logging.Logger) -> tuple[Any, str]:
    # [FUNCTION DOC] Runs the blocking start hook in a worker thread and returns its pair of close
    # function and base url. The result is also recorded thread-side with a done event that is set
    # whether the hook succeeds or fails. If the awaiting coroutine is cancelled (the worker thread
    # cannot be cancelled), a non-daemon reaper thread named after the task is started that waits
    # for the worker to finish and then closes every recorded sandbox, logging success or a warning
    # on close failure; the cancellation is then re-raised. Errors from the hook propagate to the
    # caller unchanged (nothing is recorded then).
    """One sandbox-create attempt, safe against cancellation mid-create."""
    result: list[tuple[Any, str]] = []
    done = threading.Event()

    def _start() -> tuple[Any, str]:
        # [FUNCTION DOC] HELPER: Worker-thread body: calls the start hook, appends its result to the
        # shared record, always sets the done event, and returns the recorded pair.
        try:
            result.append(start_fn(task_id, tasks_dir))
        finally:
            done.set()
        return result[0]

    try:
        return await asyncio.to_thread(_start)
    except asyncio.CancelledError:

        def _reap() -> None:
            # [FUNCTION DOC] HELPER: Reaper thread body: waits for the create to finish, then
            # closes any sandbox it produced, logging an info line on success or a warning when
            # the close raises.
            done.wait()
            for close_fn, _url in result:
                try:
                    close_fn()
                    logger.info(f"Closed sandbox orphaned by cancelled episode: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to close orphaned sandbox for {task_id}: {e}")

        # not a daemon: a subproc agent call waits for it before its process exits
        threading.Thread(target=_reap, name=f"tb2-sandbox-reap-{task_id}", daemon=False).start()
        raise


def _agent_function():
    # [FUNCTION DOC] HELPER: Imports the agent function module on first use and returns the module
    # object, so that callers needing only the registry do not require the agent loop's
    # dependencies.
    """``openenv_agent_function``, imported on use rather than at module scope.

    The one lazy import in this module, and the reason is the module's other
    callers: a provider CLI (``python tb2_sandbox_e2b.py ...``) and the
    launcher reach this module for the backend registry alone, and neither
    should have to have the agent loop's dependencies (openai) installed to
    resolve a backend name.
    """
    import openenv_agent_function

    return openenv_agent_function


# Throttle text every provider shares: an HTTP 429 surfaced as a message
# rather than a typed error. A backend adds its own vocabulary (Daytona's
# "throttler", Modal's "resource exhausted").
_THROTTLE_TEXT = ("too many requests", "429", "rate limit")


def throttle_text(exc: BaseException, *extra_patterns: str, patterns_env_var: str | None = None) -> bool:
    # [FUNCTION DOC] Returns true if the lowercased message of the error contains any shared
    # throttle phrase (too many requests, 429, rate limit) or any extra pattern (lowercased). If
    # not and an environment variable name is given, reads that variable at call time, splits it on
    # commas, trims and lowercases the chunks, ignores empty ones, and returns true if any chunk is
    # contained in the message. Otherwise false. Matching is plain substring containment.
    """True when *exc*'s message reads like throttling or exhausted capacity.

    The typed-exception half of the decision stays in the backend (only it knows
    its SDK's class); this is the half that is identical everywhere, plus the
    backend's *extra_patterns*.

    *patterns_env_var* extends the set at runtime, and exists for exactly one
    case: an endpoint whose capacity errors are worded by whoever deployed it,
    which is why only the e2b backend (self-hosted AgentENV) passes one.
    """
    s = str(exc).lower()
    if any(p in s for p in (*_THROTTLE_TEXT, *(p.lower() for p in extra_patterns))):
        return True
    raw = os.getenv(patterns_env_var, "") if patterns_env_var else ""
    return any(p in s for p in (chunk.strip().lower() for chunk in raw.split(",")) if p)


# Backoff shape for throttled creates: jittered exponential, shared by every
# backend. Constants rather than knobs — a rollout tunes how MANY creates are in
# flight (create_concurrency) and how long to keep trying (max_retries), not the
# curve between them.
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0

# Knob defaults shared by every backend, read from its OWN prefix so one
# provider can be re-tuned without touching another.
_KNOB_DEFAULTS = {
    "create_concurrency": ("CREATE_CONCURRENCY", 4, int),
    "max_retries": ("CREATE_MAX_RETRIES", 8, int),
}


def backend_knobs(env_prefix: str) -> dict[str, Any]:
    # [FUNCTION DOC] Reads the create-throttling settings for a provider prefix from the
    # environment variables named OPENENV_<prefix>_CREATE_CONCURRENCY and
    # OPENENV_<prefix>_CREATE_MAX_RETRIES (defaults 4 and 8; values converted to integers; an
    # invalid number raises a value error). Returns a mapping from the backend field names
    # (create concurrency, max retries) to those values, for use as constructor keywords.
    """The create-throttling knobs for a backend, read from OPENENV_<PREFIX>_*.

    Read once, at the importing module's import time: a rollout worker is a
    fresh process per run, so a knob changed mid-run would be a knob that
    silently did nothing.
    """
    return {
        field: cast(os.getenv(f"OPENENV_{env_prefix}_{suffix}", str(default)))
        for field, (suffix, default, cast) in _KNOB_DEFAULTS.items()
    }


@dataclass
class SandboxBackend:
    # [CLASS DOC] Dataclass holding one provider's configuration and implementing the shared
    # per-episode sandbox flow. Fields: provider display name (also used in the semaphore name and
    # log text); start_sandbox hook; is_throttle classifier; logger; create_concurrency (default 4,
    # max in-flight creates per rollout); max_retries (default 8, retries of throttled creates);
    # backoff_base_s (2.0) and backoff_cap_s (30.0) shaping the retry delay.
    """The provider-blind half of a per-episode sandbox agent function.

    Every backend is the same rollout-facing object — throttled cancel-safe
    creates, an env client bound to one episode's sandbox, the training and
    direct-drive entry points — around two provider-specific holes:
    ``start_sandbox`` (how ONE sandbox with the env server comes into being)
    and ``is_throttle`` (which of its SDK's errors are worth retrying). A backend
    module supplies those two, its knobs, and its documentation; nothing else
    about it can drift from its siblings because nothing else lives there.

    The fields are the seams: tests substitute ``start_sandbox`` or
    ``episode_env`` and shrink the backoff on the instance, so the knobs stay
    patchable without a backend re-reading its environment.
    """

    provider: str
    start_sandbox: StartFn
    is_throttle: Callable[[BaseException], bool]
    logger: logging.Logger
    create_concurrency: int = _KNOB_DEFAULTS["create_concurrency"][1]
    max_retries: int = _KNOB_DEFAULTS["max_retries"][1]
    backoff_base_s: float = BACKOFF_BASE_S
    backoff_cap_s: float = BACKOFF_CAP_S

    async def start_task_sandbox(self, task_id: str) -> tuple[Any, str]:
        # [FUNCTION DOC] Creates a sandbox for the task and returns the close function and base
        # url. Reads the task directory from OPENENV_TB2_TASKS_DIR (trimmed, may be empty) at call
        # time. Loop: acquire the per-rollout create semaphore (named by provider, limited by the
        # create concurrency) only for the duration of one cancel-safe create attempt. On an
        # exception: re-raise if it is not classified as throttling or the retry count is already
        # at max_retries; else increment the attempt, compute a delay equal to the smaller of the
        # cap and base times two to the (attempt - 1), multiplied by a random factor in [0.5, 1.5),
        # log a warning, sleep (semaphore released meanwhile) and try again.
        """Create one sandbox for *task_id* with the env server running.

        Returns (close_fn, base_url); close_fn releases the sandbox. Creation
        is throttled per rollout (the create semaphore) and retried with
        jittered exponential backoff on throttle errors; anything else
        propagates immediately.
        """
        tasks_dir = os.getenv("OPENENV_TB2_TASKS_DIR", "").strip()
        attempt = 0
        while True:
            try:
                async with rollout_semaphore(f"openenv-{self.provider}-create", self.create_concurrency):
                    return await create_once(self.start_sandbox, task_id, tasks_dir, logger=self.logger)
            except Exception as e:
                if not self.is_throttle(e) or attempt >= self.max_retries:
                    raise
                attempt += 1
                delay = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1))) * (0.5 + random.random())
                self.logger.warning(
                    f"{self.provider} create throttled for {task_id} "
                    f"(attempt {attempt}/{self.max_retries}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    @asynccontextmanager
    async def episode_env(self, env_cls: Any, metadata: dict[str, Any]):
        # [FUNCTION DOC] Async context manager for one episode's lifetime. Takes the task id from
        # metadata ("task_id", falling back to "task_name"); raises a value error if neither is
        # set. Starts a sandbox, connects an env client to its url (with the message timeout from
        # the agent function module) and yields the connected client. In all exits (including
        # errors and cancellation) it closes the sandbox in a worker thread; a failing close is
        # only logged as a warning.
        """Yield a connected env client on a fresh sandbox; release it after."""
        oaf = _agent_function()
        task_id = metadata.get("task_id") or metadata.get("task_name")
        if not task_id:
            raise ValueError("the sandbox is built for one task: metadata['task_id'] is required")
        close_fn, url = await self.start_task_sandbox(str(task_id))
        try:
            async with env_cls(base_url=url, message_timeout_s=oaf.MESSAGE_TIMEOUT_S) as env:
                yield env
        finally:
            try:
                await asyncio.to_thread(close_fn)
            except Exception as e:
                self.logger.warning(f"Failed to close sandbox for {task_id}: {e}")

    async def _run_body(self, env_cls: Any, metadata: dict[str, Any], body: Any) -> Any:
        # [FUNCTION DOC] HELPER: The env-provisioning strategy passed to the shared agent loop: runs
        # the body inside this instance's episode env context and returns its result.
        # self.episode_env, not the module function: an instance attribute
        # substituted by a test must win here.
        async with self.episode_env(env_cls, metadata) as env:
            return await body(env)

    async def run_episode(
        self,
        policy: Any,
        model_name: str,
        messages: list[dict[str, str]],
        request_kwargs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[float | None, dict[str, Any]]:
        # [FUNCTION DOC] Direct-drive entry: runs the shared multi-turn loop (loading the env
        # classes from the agent function module) with this backend's per-episode sandbox strategy
        # and no post-episode hook, using the caller's policy client. Returns the loop's reward
        # (none for no verdict) and metrics; timeouts and errors are the caller's concern.
        """One episode in its own sandbox, with the caller's own policy.
        Direct-drive entry, same contract as openenv_agent_function's.

        No post-episode hygiene: the sandbox is released when the episode ends.
        """
        oaf = _agent_function()
        return await oaf.multi_turn(
            oaf.load_tbench2(),
            policy,
            model_name,
            messages,
            request_kwargs,
            metadata,
            run_body=self._run_body,
        )

    async def run(
        self,
        base_url: str,
        prompt: Any,
        request_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        # [FUNCTION DOC] Miles agent entry point: delegates to the shared training wrapper (session
        # url policy client, wall-clock cap, failure semantics) using this backend's episode
        # function. Returns the result mapping or none to drop the sample. Extra keyword arguments
        # are ignored.
        """Run one OpenEnv tbench2 episode in its own sandbox (training entry)."""
        oaf = _agent_function()
        return await oaf.run_for_training(base_url, prompt, request_kwargs, metadata, self.run_episode)
