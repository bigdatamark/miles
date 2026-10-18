# [MODULE DOC]
# Offline unit tests for the Modal-specific part of the openenv sandbox agent function module
# (imported by bare name): its sandbox start hook (per-task sandbox from the tasks dir joined with
# the task id, closed through the materializer's complete close function) and its throttle error
# classification (text patterns such as 429, too many requests, resource exhausted, plus the sdk's
# typed resource-exhausted error when importable; other typed errors are not throttling).
"""Offline unit tests for the Modal-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is Modal's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as capacity limits, and that the start hook wires the
materialization (per-task dir, the build+create deadline, terminate-on-close).
"""

import sys
import types
from pathlib import Path

import openenv_modal_agent_function as omaf
import pytest

# --- start hook -------------------------------------------------------------


def test_start_hook_creates_per_task_and_uses_materializer_cleanup(monkeypatch):
    # [FUNCTION DOC] Replaces the Modal module's sandbox creation with a fake returning an opaque
    # sandbox object and a url, and its close function with a recorder. Calls the start hook for
    # task regex-chess under /tasks. Asserts the returned url, task dir /tasks/regex-chess, no
    # extra keyword arguments, and that the close function passes exactly that sandbox object to
    # the module's close function.
    """The sandbox is per-task (tasks_dir/task_id), the create carries the
    materializer's defaults, and close_fn uses its complete cleanup contract."""
    created = {}
    closed = []

    sandbox = object()

    def fake_create(task_dir, **kwargs):
        # [FUNCTION DOC] HELPER: Records the task dir and keyword arguments; returns the sandbox object and a url string.
        created.update(task_dir=task_dir, kwargs=kwargs)
        return sandbox, "https://abc.r5.modal.host"

    monkeypatch.setattr(omaf.tb2_sandbox_modal, "create_task_sandbox", fake_create)
    monkeypatch.setattr(omaf.tb2_sandbox_modal, "close_sandbox", closed.append)

    close_fn, url = omaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://abc.r5.modal.host"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    close_fn()
    assert closed == [sandbox]


# --- throttle classification ------------------------------------------------


def test_is_throttle_error_classification():
    # [FUNCTION DOC] Asserts HTTP 429, too-many-requests and a resource-exhausted container
    # concurrency message are throttling and an image build failure is not.
    assert omaf._is_throttle_error(Exception("HTTP 429"))
    assert omaf._is_throttle_error(Exception("Too Many Requests"))
    # This backend's own vocabulary: a hit container-concurrency ceiling.
    assert omaf._is_throttle_error(Exception("RESOURCE EXHAUSTED: too many containers"))
    assert not omaf._is_throttle_error(RuntimeError("image build failed"))


def test_is_throttle_error_typed_modal_class():
    # [FUNCTION DOC] Skipped when the modal exception module is not importable. Asserts the sdk's
    # resource-exhausted error is throttling without keywords in its message, and its auth error is not.
    """The SDK's typed capacity error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    ex = pytest.importorskip("modal.exception")
    assert omaf._is_throttle_error(ex.ResourceExhaustedError("no capacity"))
    assert not omaf._is_throttle_error(ex.AuthError("bad token"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    # [FUNCTION DOC] Puts an empty placeholder in the module table for the modal exception module;
    # asserts text-based classification still accepts a 429 message and rejects an image build failure.
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.setitem(sys.modules, "modal.exception", types.ModuleType("modal.exception"))
    assert omaf._is_throttle_error(Exception("HTTP 429"))
    assert not omaf._is_throttle_error(RuntimeError("image build failed"))
