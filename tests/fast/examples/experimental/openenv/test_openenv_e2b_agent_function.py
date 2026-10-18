# [MODULE DOC]
# Offline unit tests for the E2B-specific part of the openenv sandbox agent function module
# (imported by bare name): its sandbox start hook (per-task sandbox from the tasks dir joined with
# the task id, closed by killing the sandbox) and its throttle error classification: built-in text
# patterns (rate limit, 429, too many requests), extra patterns from a comma-separated environment
# variable matched case-insensitively as substrings, and the sdk's typed rate-limit exception when
# importable (other typed exceptions are not throttling).
"""Offline unit tests for the E2B-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is E2B's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling — including the env-knob extension that
lets a self-hosted AgentENV's capacity errors be retried without a code change
— and that the start hook wires the materialization correctly.
"""

import sys
import types
from pathlib import Path

import openenv_e2b_agent_function as oeaf
import pytest


class _Throttled(Exception):
    # [CLASS DOC] HELPER: Fake exception whose text looks like a rate-limit error.
    def __str__(self):
        # [FUNCTION DOC] HELPER: Returns the fixed rate-limit message text.
        return "rate limit exceeded, please retry"


# --- start hook -------------------------------------------------------------


def test_start_hook_creates_per_task_and_closes_by_killing(monkeypatch):
    # [FUNCTION DOC] Replaces the E2B module's sandbox creation with a fake returning a handle and
    # a url, and its kill function with a recorder. Calls the start hook for task regex-chess under
    # /tasks. Asserts the returned url, task dir /tasks/regex-chess, no extra keyword arguments,
    # and that the close function kills exactly the handle.
    """The sandbox is per-task (tasks_dir/task_id) and close_fn kills it — the
    template stays, the sandbox must not."""
    killed = []
    created = {}

    def fake_create(task_dir, **kwargs):
        # [FUNCTION DOC] HELPER: Records the task dir and keyword arguments; returns a handle string and a url string.
        created.update(task_dir=task_dir, kwargs=kwargs)
        return "sandbox-handle", "https://8000-abc.e2b.app"

    monkeypatch.setattr(oeaf.tb2_sandbox_e2b, "create_task_sandbox", fake_create)
    monkeypatch.setattr(oeaf.tb2_sandbox_e2b, "kill_sandbox", killed.append)

    close_fn, url = oeaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://8000-abc.e2b.app"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    close_fn()
    assert killed == ["sandbox-handle"]


# --- throttle classification ------------------------------------------------


def test_is_throttle_error_classification(monkeypatch):
    # [FUNCTION DOC] With the extra patterns variable unset, asserts the fake rate-limit
    # exception, an HTTP 429 message and a too-many-requests message are throttling and a template
    # build failure is not.
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    assert oeaf._is_throttle_error(_Throttled())
    assert oeaf._is_throttle_error(Exception("HTTP 429"))
    assert oeaf._is_throttle_error(Exception("Too Many Requests"))
    assert not oeaf._is_throttle_error(RuntimeError("template build failed"))


def test_is_throttle_error_extra_patterns_env(monkeypatch):
    # [FUNCTION DOC] Without the variable, a no-sufficient-capacity message is not throttling.
    # After setting the variable to a comma-and-space separated list of two patterns, that message
    # and an upper-case at-capacity message are throttling (case-insensitive, entries trimmed),
    # while the template build failure still is not.
    """OPENENV_E2B_THROTTLE_PATTERNS extends the retryable set for
    provider-specific capacity errors (e.g. a full self-hosted AgentENV pool)
    without a code change."""
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    assert not oeaf._is_throttle_error(Exception("no node with sufficient capacity"))
    monkeypatch.setenv("OPENENV_E2B_THROTTLE_PATTERNS", "sufficient capacity, at capacity")
    assert oeaf._is_throttle_error(Exception("no node with sufficient capacity"))
    assert oeaf._is_throttle_error(Exception("cluster AT CAPACITY"))
    assert not oeaf._is_throttle_error(RuntimeError("template build failed"))


def test_is_throttle_error_typed_e2b_class(monkeypatch):
    # [FUNCTION DOC] Skipped when the e2b exceptions module is not importable. Asserts the sdk's
    # rate-limit exception is throttling even without keywords in its message, and its
    # authentication exception is not.
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    ex = pytest.importorskip("e2b.exceptions")
    assert oeaf._is_throttle_error(ex.RateLimitException("slow down"))
    assert not oeaf._is_throttle_error(ex.AuthenticationException("bad key"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    # [FUNCTION DOC] Puts an empty placeholder in the module table for the e2b exceptions module;
    # asserts text-based classification still accepts a 429 message and rejects a template build failure.
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    monkeypatch.setitem(sys.modules, "e2b.exceptions", types.ModuleType("e2b.exceptions"))
    assert oeaf._is_throttle_error(Exception("HTTP 429"))
    assert not oeaf._is_throttle_error(RuntimeError("template build failed"))
