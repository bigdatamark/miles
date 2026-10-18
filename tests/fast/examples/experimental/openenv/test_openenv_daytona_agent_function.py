# [MODULE DOC]
# Offline unit tests for the Daytona-specific part of the openenv sandbox agent function module
# (imported by bare name): its sandbox start hook (per-task sandbox from the tasks dir joined with
# the task id, closed by deleting the sandbox through the client) and its throttle error
# classification (text patterns such as 429, throttler, quota or limit exceeded messages, plus the
# sdk's typed rate-limit error when importable).
"""Offline unit tests for the Daytona-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is Daytona's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling, and that the start hook wires the
materialization (client, per-task dir, delete-on-close) correctly.
"""

import sys
import types
from pathlib import Path

import openenv_daytona_agent_function as odaf
import pytest


class _Throttled(Exception):
    # [CLASS DOC] HELPER: Fake exception whose text looks like a throttler too-many-requests error.
    def __str__(self):
        # [FUNCTION DOC] HELPER: Returns the fixed throttler message text.
        return "ThrottlerException: Too Many Requests"


# --- start hook -------------------------------------------------------------


def test_start_hook_creates_per_task_and_closes_by_deleting(monkeypatch):
    # [FUNCTION DOC] Replaces the Daytona module's client factory with one returning a fake client
    # and its sandbox creation with a fake returning a handle and a preview url. Calls the start
    # hook for task regex-chess under /tasks. Asserts the returned url is the preview url; creation
    # got the client and the task dir /tasks/regex-chess and no extra keyword arguments; calling
    # the returned close function deletes exactly the sandbox handle through the client.
    """The sandbox is per-task (tasks_dir/task_id) and close_fn must DELETE it:
    a stopped-but-kept Daytona sandbox still occupies the org's quota."""
    deleted = []
    created = {}

    class _FakeDaytona:
        # [CLASS DOC] HELPER: Fake Daytona client recording deleted sandboxes.
        def delete(self, sandbox):
            # [FUNCTION DOC] HELPER: Appends the sandbox to the deleted list.
            deleted.append(sandbox)

    client = _FakeDaytona()

    def fake_create(daytona, task_dir, **kwargs):
        # [FUNCTION DOC] HELPER: Records the client, task dir and extra keyword arguments; returns
        # a sandbox handle string and a preview url string.
        created.update(daytona=daytona, task_dir=task_dir, kwargs=kwargs)
        return "sandbox-handle", "https://preview.daytona/8000"

    monkeypatch.setattr(odaf.tb2_sandbox_daytona, "make_daytona", lambda: client)
    monkeypatch.setattr(odaf.tb2_sandbox_daytona, "create_task_sandbox", fake_create)

    close_fn, url = odaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://preview.daytona/8000"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    assert created["daytona"] is client
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    close_fn()
    assert deleted == ["sandbox-handle"]


# --- throttle classification ------------------------------------------------


def test_is_throttle_error_classification():
    # [FUNCTION DOC] Asserts the throttle classifier accepts the fake throttler exception, a
    # message with HTTP 429, and a message containing throttler; and rejects a runtime error about
    # an image build failure.
    assert odaf._is_throttle_error(_Throttled())
    assert odaf._is_throttle_error(Exception("HTTP 429"))
    assert odaf._is_throttle_error(Exception("throttler tripped"))  # this backend's own vocabulary
    assert not odaf._is_throttle_error(RuntimeError("image build failed"))


def test_is_throttle_error_quota_exhaustion():
    # [FUNCTION DOC] Asserts total cpu limit exceeded and total memory limit exceeded messages are
    # throttle errors, while a runtime error about a missing docker image in the task file is not.
    """Org capacity errors are retryable: failing immediately aborts the episode
    and discards the whole group."""
    assert odaf._is_throttle_error(Exception("Failed to create sandbox: Total CPU limit exceeded. Max allowed: 500."))
    assert odaf._is_throttle_error(Exception("Failed to create sandbox: Total memory limit exceeded."))
    assert not odaf._is_throttle_error(RuntimeError("task.toml is missing docker_image"))


def test_is_throttle_error_typed_daytona_class():
    # [FUNCTION DOC] Skipped when the daytona errors module is not importable. Asserts the sdk's
    # rate-limit error class, with a message lacking throttle keywords, is classified as throttling.
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords."""
    errors = pytest.importorskip("daytona.common.errors")
    assert odaf._is_throttle_error(errors.DaytonaRateLimitError("slow down"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    # [FUNCTION DOC] Puts an empty placeholder in the module table for the daytona errors module
    # (so the typed class is missing); asserts text-based classification still accepts a 429
    # message and rejects an image build failure.
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.setitem(sys.modules, "daytona.common.errors", types.ModuleType("daytona.common.errors"))
    assert odaf._is_throttle_error(Exception("HTTP 429"))
    assert not odaf._is_throttle_error(RuntimeError("image build failed"))
