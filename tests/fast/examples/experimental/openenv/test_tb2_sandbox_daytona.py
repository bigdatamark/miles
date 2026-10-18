# [MODULE DOC]
# Tests for the Daytona sandbox materialization module of the openenv terminal-bench-2 example
# (imported by bare name as sandbox): ownership labels attached to sandboxes (task name from the
# task dir, launcher identity from an environment variable or fallback, optional run id label), a
# single shared cached client, api key resolution (environment value, then key file path from an
# environment variable, then a default file under the home config folder, else a runtime error),
# the dead-man's-switch contract that creates arm auto-stop and auto-delete timers by default, and
# the background keepalive thread that refreshes sandbox activity at an interval and exits after a
# persistent failure.
"""Tests for the Daytona sandbox half: ownership labels and the orphan-TTL
keepalive lifecycle.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q
"""

import inspect
import sys
import threading
import time
from pathlib import Path

import pytest

import tb2_sandbox_daytona as sandbox


def test_sandbox_labels_default_to_unix_user(monkeypatch):
    # [FUNCTION DOC] With the launcher and run id variables unset, the labels for a task dir
    # /opt/tb2-tasks/regex-chess hold the task name regex-chess, a non-empty launcher identity, and
    # no run id label at all (omitted rather than empty).
    monkeypatch.delenv("OPENENV_LAUNCHER", raising=False)
    monkeypatch.delenv("OPENENV_RUN_ID", raising=False)
    labels = sandbox.sandbox_labels(Path("/opt/tb2-tasks/regex-chess"))
    assert labels["openenv-tbench2-task"] == "regex-chess"
    assert labels["openenv-launcher"]  # some non-empty identity, never absent
    assert "openenv-run-id" not in labels  # omitted when unset, not ""


def test_sandbox_labels_explicit_launcher_and_run_id(monkeypatch):
    # [FUNCTION DOC] With launcher and run id variables set, the labels carry exactly those values.
    monkeypatch.setenv("OPENENV_LAUNCHER", "tao-lin")
    monkeypatch.setenv("OPENENV_RUN_ID", "tb2-grpo-0717")
    labels = sandbox.sandbox_labels(Path("/opt/tb2-tasks/regex-chess"))
    assert labels["openenv-launcher"] == "tao-lin"
    assert labels["openenv-run-id"] == "tb2-grpo-0717"


def test_make_daytona_reuses_one_client(monkeypatch):
    # [FUNCTION DOC] Clears the module's cached client, sets a test api key, installs a fake
    # daytona module whose client class records its config and whose config builder returns its
    # keyword arguments. Calls the client factory five times; asserts the client class was
    # constructed once and every call returned the same object.
    """One client per call is a socket leak: the SDK owns a connection pool with
    no close(), and this runs once per create attempt, retries included."""
    monkeypatch.setattr(sandbox, "_client", None)
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn_test")
    built = []

    class _Daytona:
        # [CLASS DOC] HELPER: Fake Daytona client recording the config it was built with.
        def __init__(self, config):
            # [FUNCTION DOC] HELPER: Appends the config to the built list.
            built.append(config)

    fake = type(sys)("daytona")
    fake.Daytona = _Daytona
    fake.DaytonaConfig = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "daytona", fake)

    clients = [sandbox.make_daytona() for _ in range(5)]
    assert len(built) == 1, "make_daytona built a client per call"
    assert all(c is clients[0] for c in clients)


def test_resolve_api_key_env_value_wins(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With both the key variable and a key file variable set, the resolved key is
    # the variable's value, not the file's.
    key_file = tmp_path / "api_key"
    key_file.write_text("dtn_from_file\n")
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn_from_env")
    monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "dtn_from_env"


def test_resolve_api_key_falls_back_to_file(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With only the key file path variable set, the resolved key is the file's
    # content with surrounding whitespace stripped.
    # The launcher forwards only this path (never the value, which would be
    # echoed into driver logs via ray runtime_env); workers must read it here.
    key_file = tmp_path / "api_key"
    key_file.write_text("dtn_from_file\n")
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "dtn_from_file"  # whitespace stripped


def test_resolve_api_key_default_path_under_home(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With neither variable set and the home variable pointing at a temp dir
    # holding the key file in the daytona folder under the config folder, the resolved key is that file's content.
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".config" / "daytona"
    cfg.mkdir(parents=True)
    (cfg / "api_key").write_text("dtn_default\n")
    assert sandbox.resolve_api_key() == "dtn_default"


def test_resolve_api_key_errors_when_absent(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With no key variable and a key file path naming a missing file, resolution
    # raises a runtime error saying the key is missing or empty.
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing or empty"):
        sandbox.resolve_api_key()


def test_create_arms_ttl_by_default():
    # [FUNCTION DOC] Inspects the signature of the task sandbox creation function and asserts its
    # auto-stop minutes and auto-delete minutes parameters both default to positive numbers.
    # The dead-man's-switch contract: creates must arm auto-stop/auto-delete,
    # or a hard-killed caller's orphans run (and bill) forever.
    sig = inspect.signature(sandbox.create_task_sandbox)
    assert sig.parameters["auto_stop_minutes"].default > 0
    assert sig.parameters["auto_delete_minutes"].default > 0


def test_keepalive_beats_then_exits_on_persistent_failure(monkeypatch):
    # [FUNCTION DOC] Sets the keepalive interval to 0.02 seconds and starts the keepalive on a
    # stub sandbox whose activity refresh counts beats and raises once marked dead. Waits (up to 2
    # seconds) for at least 3 beats. Then marks the stub dead and waits (up to 2 seconds) until no
    # live thread has keepalive in its name; asserts none remains.
    monkeypatch.setattr(sandbox, "_KEEPALIVE_INTERVAL_S", 0.02)

    class Stub:
        # [CLASS DOC] HELPER: Fake sandbox counting activity refreshes and failing after being marked dead.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Starts with zero beats and alive.
            self.beats = 0
            self.dead = False

        def refresh_activity(self):
            # [FUNCTION DOC] HELPER: Raises a runtime error if dead; otherwise increments the beat count.
            if self.dead:
                raise RuntimeError("sandbox deleted")
            self.beats += 1

    stub = Stub()
    sandbox._start_keepalive(stub, "regex-chess")
    deadline = time.time() + 2.0
    while stub.beats < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert stub.beats >= 3  # beats while the sandbox is alive

    stub.dead = True  # episode over, sandbox deleted -> thread must exit
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any("keepalive" in t.name for t in threading.enumerate()):
            break
        time.sleep(0.01)
    assert not any("keepalive" in t.name for t in threading.enumerate())
