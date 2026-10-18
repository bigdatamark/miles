# [MODULE DOC]
# Tests for the Modal sandbox materialization module of the openenv terminal-bench-2 example
# (imported by bare name as sandbox), run against a fake modal sdk installed in the module table
# (image with recorded layers, tunnel, sandbox, sandbox class, readiness probe, app lookup) with
# the recipe pieces (image resolver, server layer commands, labels, server command, readiness wait,
# task resources) patched. Covered: one image layer per recipe command and anonymous registry
# pulls; sandbox sizing from the task config file with floors and no disk; the app handle looked up
# once per process; the create path (server as entrypoint command, encrypted port, lifetime and
# idle timeouts from arguments or module defaults, tags, tcp readiness probe, cpu and memory,
# readiness wait, termination plus detach on any failure, diagnostics from an exited server's
# output but never reading a live sandbox's stream, a wall clock timeout around creation); closing a
# sandbox (terminate then detach even when termination fails); and the tunnel url for the server port.
"""Tests for the Modal sandbox half: image layering, sandbox sizing, the
tunnel URL, and the create path's lifetime/diagnostic contract.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q
"""

import sys
import threading
import types
from pathlib import Path

import pytest

import tb2_sandbox_modal as sandbox

_COMMANDS = ("apt-get install -y curl", "curl -LsSf https://astral.sh/uv/install.sh | sh", "mkdir -p /opt/tb2-tasks")


# --- fake modal SDK ---------------------------------------------------------


class _FakeImage:
    # [CLASS DOC] HELPER: Fake modal image holding a tag and a list of layers, each layer being the
    # tuple of commands of one run-commands call (calls return new images, leaving the old one
    # unchanged). The registry constructor records its keyword arguments in a class-level attribute.
    """Records layering: every run_commands call is one layer."""

    def __init__(self, tag, layers=()):
        # [FUNCTION DOC] HELPER: Stores the tag and a list copy of the layers.
        self.tag = tag
        self.layers = list(layers)

    def run_commands(self, *commands):
        # [FUNCTION DOC] HELPER: Returns a new image with the same tag and the commands tuple appended as one more layer.
        return _FakeImage(self.tag, [*self.layers, commands])

    @staticmethod
    def from_registry(tag, **kwargs):
        # [FUNCTION DOC] HELPER: Records the keyword arguments in the class-level attribute and returns a layerless image with the tag.
        _FakeImage.from_registry_kwargs = kwargs
        return _FakeImage(tag)


class _FakeTunnel:
    # [CLASS DOC] HELPER: Fake tunnel exposing a url.
    def __init__(self, url):
        # [FUNCTION DOC] HELPER: Stores the url.
        self.url = url


class _FakeSandbox:
    # [CLASS DOC] HELPER: Stands in for one created sandbox. Holds an exit code reported by poll
    # (none means still running), an output string returned by the stdout read, flags for terminated
    # and detached, the timeouts it was last asked to use for readiness and tunnels, and a tunnel url.
    """Stands in for one created sandbox."""

    def __init__(self, exit_code=None, output="", tunnel_url="https://abc.r5.modal.host"):
        # [FUNCTION DOC] HELPER: Stores the exit code and tunnel url, builds a stdout object whose
        # read returns the output, and initializes the flags and recorded timeouts.
        self._exit_code = exit_code
        self.stdout = types.SimpleNamespace(read=lambda: output)
        self.terminated = False
        self.detached = False
        self.ready_timeout = None
        self.tunnel_timeout = None
        self._tunnel_url = tunnel_url

    def wait_until_ready(self, *, timeout=300):
        # [FUNCTION DOC] HELPER: Records the timeout (default 300) as the ready timeout.
        self.ready_timeout = timeout

    def tunnels(self, timeout=50):
        # [FUNCTION DOC] HELPER: Records the timeout (default 50) and returns a mapping from port
        # 8000 to a tunnel with the sandbox's url.
        self.tunnel_timeout = timeout
        return {8000: _FakeTunnel(self._tunnel_url)}

    def terminate(self):
        # [FUNCTION DOC] HELPER: Marks the sandbox terminated.
        self.terminated = True

    def detach(self):
        # [FUNCTION DOC] HELPER: Marks the sandbox detached.
        self.detached = True

    def poll(self):
        # [FUNCTION DOC] HELPER: Returns the exit code, or none while still running.
        return self._exit_code


class _FakeSandboxCls:
    # [CLASS DOC] HELPER: Fake modal sandbox class with class-level state: the arguments of the
    # last create call and the instance that create returns.
    created = None
    instance = None

    @staticmethod
    def create(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Records the positional and keyword arguments in the created
        # attribute and returns the class-level instance.
        _FakeSandboxCls.created = {"args": args, "kwargs": kwargs}
        return _FakeSandboxCls.instance


class _FakeProbe:
    # [CLASS DOC] HELPER: Fake modal readiness probe factory.
    @staticmethod
    def with_tcp(port, **kwargs):
        # [FUNCTION DOC] HELPER: Returns a tuple of the word tcp, the port and the keyword arguments.
        return ("tcp", port, kwargs)


class _FakeApp:
    # [CLASS DOC] HELPER: Fake modal app class with a class-level list of lookups (set by the fake modal fixture).
    lookups = None

    @staticmethod
    def lookup(name, **kwargs):
        # [FUNCTION DOC] HELPER: Appends the name and keyword arguments to the lookups list and returns app: followed by the name.
        _FakeApp.lookups.append((name, kwargs))
        return f"app:{name}"


@pytest.fixture
def fake_modal(monkeypatch):
    # [FUNCTION DOC] HELPER: Resets the fake app lookups and sets the fake sandbox class instance
    # to a fresh default sandbox. Builds and installs a fake modal module (image, probe, sandbox,
    # app) in the module table, resets the module's cached app to none, and patches: image resolver
    # to return a fixed task image tag; server layer commands to return the constant command list;
    # labels to a mapping with the task dir name; server command to a string serve followed by the
    # default task id; readiness wait to a no-op; task resources to 2.0 cpus and 4096 memory.
    # Returns the fake module.
    """Install a fake `modal` module and reset the backend's process-wide app cache."""
    _FakeApp.lookups = []
    _FakeSandboxCls.instance = _FakeSandbox()
    mod = types.ModuleType("modal")
    mod.Image = _FakeImage
    mod.Probe = _FakeProbe
    mod.Sandbox = _FakeSandboxCls
    mod.App = _FakeApp
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setattr(sandbox, "_app", None)
    monkeypatch.setattr(sandbox, "resolve_docker_image", lambda task_dir, override=None: "ghcr.io/laude/task:1.0")
    monkeypatch.setattr(sandbox, "server_layer_commands", lambda task_dir: list(_COMMANDS))
    monkeypatch.setattr(sandbox, "sandbox_labels", lambda task_dir: {"openenv-tbench2-task": task_dir.name})
    monkeypatch.setattr(sandbox, "server_cmd", lambda timeout, default_task_id="": f"serve {default_task_id}")
    monkeypatch.setattr(sandbox, "wait_server_ready", lambda url, timeout_s=300.0: None)
    monkeypatch.setattr(sandbox, "task_resources", lambda task_dir: {"cpu": 2.0, "memory": 4096})
    return mod


# --- image layering ---------------------------------------------------------


def test_task_image_is_one_layer_per_recipe_command(fake_modal):
    # [FUNCTION DOC] Asserts the task image has the resolved image tag and exactly one layer (a
    # one-element tuple) per recipe command, in order.
    """Per-command layers so that editing the recipe's tail re-runs only the
    layers below the edit; collapsing the recipe into one layer would reinstall
    apt/uv for every such change. (The saving is per task: a task's own base
    image makes its chain hash unique, so nothing is shared across tasks.)"""
    image = sandbox.task_image(Path("/tasks/regex-chess"))
    assert image.tag == "ghcr.io/laude/task:1.0"
    assert image.layers == [(c,) for c in _COMMANDS]


def test_task_image_pulls_anonymously_by_default(fake_modal):
    # [FUNCTION DOC] Builds a task image and asserts the registry constructor received no keyword
    # arguments (no registry secret).
    """No registry secret is passed: the TB2 task images are public, and a
    private one would have to be opted into explicitly."""
    sandbox.task_image(Path("/tasks/t"))
    assert _FakeImage.from_registry_kwargs == {}


# --- sandbox sizing ---------------------------------------------------------


def test_task_resources_floors_and_omits_disk(tmp_path):
    # [FUNCTION DOC] With the real (unpatched) function: a task config with 4 cpus, 8192 megabytes
    # and a storage setting yields cpu 4.0 and memory 8192 only (no disk key); an empty environment
    # section yields the floors cpu 1.0 and memory 2048. Note the fake modal fixture is not used,
    # so the module's real resource function runs.
    """Scalar requests track task.toml without adding a hard burst limit;
    storage_mb has no Modal counterpart and must not leak into create."""
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 4\nmemory_mb = 8192\nstorage_mb = 20480\n")
    assert sandbox.task_resources(tmp_path) == {"cpu": 4.0, "memory": 8192}

    (tmp_path / "task.toml").write_text("[environment]\n")
    assert sandbox.task_resources(tmp_path) == {"cpu": 1.0, "memory": 2048}


# --- app handle -------------------------------------------------------------


def test_app_is_looked_up_once_per_process(fake_modal, monkeypatch):
    # [FUNCTION DOC] Sets the app name (environment variable and module constant) to
    # openenv-tbench2, asks for the app handle twice; asserts both calls return app:openenv-tbench2
    # and the fake lookup was called once with create-if-missing true.
    """App.lookup is a network round-trip and a rollout fans out many episodes."""
    monkeypatch.setenv("OPENENV_MODAL_APP", "openenv-tbench2")
    monkeypatch.setattr(sandbox, "_APP_NAME", "openenv-tbench2")
    assert sandbox.get_app() == "app:openenv-tbench2"
    assert sandbox.get_app() == "app:openenv-tbench2"
    assert _FakeApp.lookups == [("openenv-tbench2", {"create_if_missing": True})]


# --- create path ------------------------------------------------------------


def test_create_passes_the_lifetime_and_ownership_contract(fake_modal):
    # [FUNCTION DOC] Creates the task sandbox for regex-chess with command timeout 600, lifetime
    # 1200 and idle timeout 120 seconds. Asserts: returned url is the tunnel url and the returned
    # sandbox is the fake instance; the create call's positional args are bash, -c and the serve
    # command with the task id; keyword arguments have encrypted ports [8000], timeout 1200, idle
    # timeout 120, tags equal to the labels, a tcp readiness probe on 8000 with no extra options,
    # cpu 2.0 and memory 4096; the readiness wait used 300 seconds; the sandbox was not terminated.
    sandbox_obj, url = sandbox.create_task_sandbox(
        Path("/tasks/regex-chess"), command_timeout_s=600, ttl_s=1200, idle_timeout_s=120
    )
    created = _FakeSandboxCls.created
    kwargs = created["kwargs"]

    assert url == "https://abc.r5.modal.host"
    assert sandbox_obj is _FakeSandboxCls.instance
    # The env server is the sandbox's entrypoint, not a follow-up exec: the task
    # image's own CMD must not decide what runs here.
    assert created["args"] == ("bash", "-c", "serve regex-chess")
    assert kwargs["encrypted_ports"] == [8000]
    assert kwargs["timeout"] == 1200
    assert kwargs["idle_timeout"] == 120
    assert kwargs["tags"] == {"openenv-tbench2-task": "regex-chess"}
    assert kwargs["readiness_probe"] == ("tcp", 8000, {})
    assert kwargs["cpu"] == 2.0 and kwargs["memory"] == 4096
    assert sandbox_obj.ready_timeout == 300
    assert not sandbox_obj.terminated


def test_create_defaults_lifetime_from_the_env_knobs(fake_modal, monkeypatch):
    # [FUNCTION DOC] Patches the module's lifetime constant to 1800 and idle constant to 300
    # seconds, creates with no overrides, and asserts the create call's timeout and idle timeout
    # equal those values.
    monkeypatch.setattr(sandbox, "_SANDBOX_TTL_S", 1800)
    monkeypatch.setattr(sandbox, "_IDLE_TIMEOUT_S", 300)
    sandbox.create_task_sandbox(Path("/tasks/t"))
    kwargs = _FakeSandboxCls.created["kwargs"]
    assert kwargs["timeout"] == 1800
    assert kwargs["idle_timeout"] == 300


def test_create_terminates_the_sandbox_when_the_server_never_comes_up(fake_modal, monkeypatch):
    # [FUNCTION DOC] The fake sandbox has already exited with code 1 and output mentioning a
    # missing module; the server readiness wait raises a runtime error. Asserts creation raises a
    # runtime error whose message includes that output, and the sandbox was terminated and detached.
    """A ready-timeout must not leak the sandbox, and the failure should carry
    the exited server's output instead of only a timeout."""
    _FakeSandboxCls.instance = _FakeSandbox(exit_code=1, output="Traceback: no module named tbench2_env\n")

    def boom(url, timeout_s=300.0):
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying the health check timed out.
        raise RuntimeError("health check timed out")

    monkeypatch.setattr(sandbox, "wait_server_ready", boom)

    with pytest.raises(RuntimeError, match="no module named tbench2_env"):
        sandbox.create_task_sandbox(Path("/tasks/regex-chess"))
    assert _FakeSandboxCls.instance.terminated
    assert _FakeSandboxCls.instance.detached


def test_create_cleans_up_when_the_readiness_probe_fails(fake_modal, monkeypatch):
    # [FUNCTION DOC] Makes the sandbox's readiness wait raise a runtime error; asserts creation
    # raises that error and the sandbox was terminated and detached.
    def fail_readiness(*, timeout=300):
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying readiness timed out.
        raise RuntimeError("readiness timed out")

    monkeypatch.setattr(_FakeSandboxCls.instance, "wait_until_ready", fail_readiness)

    with pytest.raises(RuntimeError, match="readiness timed out"):
        sandbox.create_task_sandbox(Path("/tasks/t"))
    assert _FakeSandboxCls.instance.terminated
    assert _FakeSandboxCls.instance.detached


def test_create_does_not_read_the_stream_of_a_live_sandbox(fake_modal, monkeypatch):
    # [FUNCTION DOC] The fake sandbox is still running (no exit code) and its stdout read raises
    # an assertion error if called; the server wait raises a runtime error. Asserts creation
    # raises the health check runtime error (stdout never read) and the sandbox was terminated and detached.
    """Reading stdout of a RUNNING sandbox would block on a stream that never
    closes, so a still-running sandbox contributes no diagnostics."""

    def exploding_read():
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying a live sandbox's stdout must not be read.
        raise AssertionError("stdout of a live sandbox must not be read")

    _FakeSandboxCls.instance = _FakeSandbox(exit_code=None)
    _FakeSandboxCls.instance.stdout = types.SimpleNamespace(read=exploding_read)

    def boom(url, timeout_s=300.0):
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying the health check timed out.
        raise RuntimeError("health check timed out")

    monkeypatch.setattr(sandbox, "wait_server_ready", boom)

    with pytest.raises(RuntimeError, match="health check timed out"):
        sandbox.create_task_sandbox(Path("/tasks/t"))
    assert _FakeSandboxCls.instance.terminated
    assert _FakeSandboxCls.instance.detached


def test_close_terminates_and_detaches(fake_modal):
    # [FUNCTION DOC] Closing the fake instance leaves it both terminated and detached.
    sandbox.close_sandbox(_FakeSandboxCls.instance)
    assert _FakeSandboxCls.instance.terminated
    assert _FakeSandboxCls.instance.detached


def test_close_detaches_when_termination_fails(fake_modal, monkeypatch):
    # [FUNCTION DOC] Makes terminate raise a runtime error; asserts closing re-raises it and the
    # sandbox was still detached.
    def fail_termination():
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying terminate failed.
        raise RuntimeError("terminate failed")

    monkeypatch.setattr(_FakeSandboxCls.instance, "terminate", fail_termination)
    with pytest.raises(RuntimeError, match="terminate failed"):
        sandbox.close_sandbox(_FakeSandboxCls.instance)
    assert _FakeSandboxCls.instance.detached


def test_create_enforces_the_build_wall_clock(fake_modal, monkeypatch):
    # [FUNCTION DOC] Replaces the fake sandbox create with one blocking on a release event for up
    # to 10 seconds; asserts creating with a create timeout of 0.05 seconds raises a timeout error.
    # The release event is always set afterwards so the blocked worker can finish.
    """The first create for a task builds the image, which Modal does not
    deadline itself; a hung build must give the slot back rather than block the
    episode forever."""
    release = threading.Event()

    def slow_create(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Waits up to 10 seconds for the release event, then returns the fake instance.
        release.wait(10)
        return _FakeSandboxCls.instance

    monkeypatch.setattr(_FakeSandboxCls, "create", staticmethod(slow_create))
    try:
        with pytest.raises(TimeoutError):
            sandbox.create_task_sandbox(Path("/tasks/t"), create_timeout_s=0.05)
    finally:
        release.set()


# --- tunnel ----------------------------------------------------------------


def test_base_url_is_the_tunnel_for_the_env_server_port(fake_modal, monkeypatch):
    # [FUNCTION DOC] Patches the tunnel timeout constant to 60 seconds and uses a fake sandbox
    # with a distinct tunnel url; asserts the base url equals that tunnel url and the tunnel
    # lookup was given the timeout 60.
    monkeypatch.setattr(sandbox, "_TUNNEL_TIMEOUT_S", 60.0)
    sb = _FakeSandbox(tunnel_url="https://xyz.r5.modal.host")
    assert sandbox.base_url(sb) == "https://xyz.r5.modal.host"
    assert sb.tunnel_timeout == 60
