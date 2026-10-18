# [MODULE DOC]
# Tests for the E2B sandbox materialization module of the openenv terminal-bench-2 example
# (imported by bare name as sandbox). Template aliasing: the alias is deterministic, lower-cased and
# sanitized, and changes when the base image, build commands, build resources or build user change
# (the recipe pieces are replaced by patched functions). Template building with a fake template
# class and a fake e2b module: skipped when the alias already exists unless forced, built with the
# recipe commands, resources, cache flag and api key, built as root with the user set before the
# first command, and bounded by a wall clock timeout. Sandbox creation with a fake sandbox: killed
# when the server never becomes ready, and the env server is started as root in the background.
# Also covers resource extraction from the task config file (with floors), api key resolution
# (environment, key file, default file, error), the base url scheme, and the sandbox lifetime and
# keepalive thread behavior.
"""Tests for the E2B sandbox half: template aliasing, key supply, and the
orphan-TTL keepalive lifecycle.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q
"""

import concurrent.futures
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import tb2_sandbox_e2b as sandbox


# --- template aliasing -------------------------------------------------------


def _patch_recipe(
    monkeypatch,
    base="ghcr.io/laude/task:1.0",
    commands=("apt-get install -y curl",),
    resources=None,
):
    # [FUNCTION DOC] HELPER: Patches the module's recipe inputs: the docker image resolver returns
    # the given base image, the server layer commands function returns the given commands as a list,
    # and the build resources function returns a copy of the given resources (default one cpu and
    # 2048 megabytes of memory). The task dir and override arguments are ignored.
    resources = resources or {"cpu_count": 1, "memory_mb": 2048}
    monkeypatch.setattr(sandbox, "resolve_docker_image", lambda task_dir, override: base)
    monkeypatch.setattr(sandbox, "server_layer_commands", lambda task_dir: list(commands))
    monkeypatch.setattr(sandbox, "task_build_resources", lambda task_dir: dict(resources))


def test_template_alias_is_deterministic_and_sanitized(monkeypatch):
    # [FUNCTION DOC] Computes the alias twice for a task dir named Regex_Chess; asserts the two
    # are equal and start with tb2-regex-chess- (lower-cased, characters outside lowercase letters,
    # digits and hyphen mapped to hyphens).
    _patch_recipe(monkeypatch)
    a1 = sandbox.template_alias(Path("/tasks/Regex_Chess"))
    a2 = sandbox.template_alias(Path("/tasks/Regex_Chess"))
    assert a1 == a2
    assert a1.startswith("tb2-regex-chess-")  # lowercased, non [a-z0-9-] mapped to '-'


def test_template_alias_tracks_recipe_content(monkeypatch):
    # [FUNCTION DOC] Asserts the alias changes when an extra build command is added, and also when
    # only the base image changes.
    """The digest must cover base image AND build commands: editing either
    re-bakes instead of silently serving a stale template."""
    _patch_recipe(monkeypatch)
    original = sandbox.template_alias(Path("/tasks/t"))
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "extra step"))
    assert sandbox.template_alias(Path("/tasks/t")) != original
    _patch_recipe(monkeypatch, base="ghcr.io/laude/task:2.0")
    assert sandbox.template_alias(Path("/tasks/t")) != original


def test_template_alias_tracks_build_resources(monkeypatch):
    # [FUNCTION DOC] Asserts the alias differs between a small and a large resource setting.
    """E2B sizes sandboxes at template-build time, so a task.toml resource
    bump must re-bake instead of warm-starting under-provisioned sandboxes."""
    _patch_recipe(monkeypatch)
    small = sandbox.template_alias(Path("/tasks/t"))
    _patch_recipe(monkeypatch, resources={"cpu_count": 4, "memory_mb": 8192})
    big = sandbox.template_alias(Path("/tasks/t"))
    assert small != big


class _FakeTemplateCls:
    # [CLASS DOC] HELPER: Fake of the e2b template builder. An instance records the base image, the
    # user, the list of run commands and an ordered list of all steps. Chaining methods return the
    # instance. The class-level exists flag answers the alias-exists query and the class-level
    # builds list collects every build call (template, alias, keyword arguments); both are set by
    # the fake e2b fixture before each test.
    # KNOWN ISSUE: the class-level instances attribute is never read or set anywhere in this file,
    # so the comment saying it is set per test is stale.
    """Stands in for e2b.Template: records build calls, controls alias_exists."""

    instances = None  # set per test

    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with empty command and step lists.
        self.commands = []
        self.steps = []  # ordered build steps, to prove set_user precedes them

    def from_image(self, base):
        # [FUNCTION DOC] HELPER: Records the base image and returns self.
        self.base = base
        return self

    def set_user(self, user):
        # [FUNCTION DOC] HELPER: Records the user and a set-user step; returns self.
        self.user = user
        self.steps.append(("set_user", user))
        return self

    def run_cmd(self, cmd):
        # [FUNCTION DOC] HELPER: Records the command and a run-command step; returns self.
        self.commands.append(cmd)
        self.steps.append(("run_cmd", cmd))
        return self

    @staticmethod
    def alias_exists(alias, **kwargs):
        # [FUNCTION DOC] HELPER: Returns the class-level exists flag (alias and keyword arguments ignored).
        return _FakeTemplateCls.exists

    @staticmethod
    def build(template, alias, **kwargs):
        # [FUNCTION DOC] HELPER: Appends the template, alias and keyword arguments to the class-level builds list.
        _FakeTemplateCls.builds.append((template, alias, kwargs))


@pytest.fixture
def fake_e2b(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture resetting the fake template class state (alias does not
    # exist, no builds), installing a fake e2b module exposing it as the template class in the
    # module table, and patching the module's key resolver to return a fake key starting with e2b_
    # followed by 40 zeros. Returns the fake module.
    _FakeTemplateCls.exists = False
    _FakeTemplateCls.builds = []
    mod = types.ModuleType("e2b")
    mod.Template = _FakeTemplateCls
    monkeypatch.setitem(sys.modules, "e2b", mod)
    monkeypatch.setattr(sandbox, "resolve_api_key", lambda: "e2b_" + "0" * 40)
    return mod


def test_ensure_template_short_circuits_on_existing_alias(monkeypatch, fake_e2b):
    # [FUNCTION DOC] With the alias reported as existing, ensuring the task template returns the
    # computed alias and performs no build.
    _patch_recipe(monkeypatch)
    _FakeTemplateCls.exists = True
    alias = sandbox.ensure_task_template(Path("/tasks/t"))
    assert alias == sandbox.template_alias(Path("/tasks/t"))
    assert _FakeTemplateCls.builds == []  # no rebuild when the alias exists


def test_ensure_template_builds_with_recipe_and_resources(monkeypatch, fake_e2b):
    # [FUNCTION DOC] With resources of 3 cpus and 4096 megabytes and no existing alias, asserts
    # exactly one build call: for the returned alias, with the template holding the base image and
    # the recipe commands, keyword arguments with cpu count 3, memory 4096, skip cache false and
    # an api key beginning with e2b_.
    _patch_recipe(monkeypatch, resources={"cpu_count": 3, "memory_mb": 4096})
    alias = sandbox.ensure_task_template(Path("/tasks/t"))
    ((template, built_alias, kwargs),) = _FakeTemplateCls.builds
    assert built_alias == alias
    assert template.base == "ghcr.io/laude/task:1.0"
    assert template.commands == ["apt-get install -y curl"]
    assert kwargs["cpu_count"] == 3 and kwargs["memory_mb"] == 4096
    assert kwargs["skip_cache"] is False
    assert kwargs["api_key"].startswith("e2b_")


def test_ensure_template_builds_as_root(monkeypatch, fake_e2b):
    # [FUNCTION DOC] Builds a two-command recipe; asserts the built template's user is root and
    # its first step is the set-user root step (before any command).
    """E2B runs build commands as a NON-root user by default, which fails every
    layer of the recipe (apt-get exits 100, /opt is not writable) — observed on
    E2B Cloud, invisible on a self-hosted AgentENV that builds as root. The user
    must be set BEFORE the first command, since set_user only affects what
    follows it."""
    _patch_recipe(monkeypatch, commands=("apt-get install -y curl", "mkdir -p /opt/envserver"))
    sandbox.ensure_task_template(Path("/tasks/t"))
    ((template, _, _),) = _FakeTemplateCls.builds
    assert template.user == "root"
    assert template.steps[0] == ("set_user", "root")


def test_template_alias_tracks_the_build_user(monkeypatch, fake_e2b):
    # [FUNCTION DOC] Asserts the alias changes when the module's build user constant is patched to a different value.
    """The user is part of the baked artifact, so changing it must re-bake
    rather than serve a template built by whoever built it first."""
    _patch_recipe(monkeypatch)
    before = sandbox.template_alias(Path("/tasks/t"))
    monkeypatch.setattr(sandbox, "_BUILD_USER", "someone-else")
    assert sandbox.template_alias(Path("/tasks/t")) != before


def test_ensure_template_force_rebuilds_skipping_cache(monkeypatch, fake_e2b):
    # [FUNCTION DOC] With the alias existing but force requested, asserts one build happened with skip cache true.
    _patch_recipe(monkeypatch)
    _FakeTemplateCls.exists = True  # force must rebuild anyway
    sandbox.ensure_task_template(Path("/tasks/t"), force=True)
    ((_, _, kwargs),) = _FakeTemplateCls.builds
    assert kwargs["skip_cache"] is True


def test_ensure_template_enforces_build_wall_clock(monkeypatch, fake_e2b):
    # [FUNCTION DOC] Replaces the fake build with one sleeping 2 seconds and requests a build
    # timeout of 0.05 seconds; asserts a futures timeout error is raised.
    """build_timeout_s bounds the whole blocking build, not one HTTP request."""
    _patch_recipe(monkeypatch)

    def hanging_build(template, alias, **kwargs):
        # [FUNCTION DOC] HELPER: Sleeps two seconds and returns nothing.
        time.sleep(2)

    monkeypatch.setattr(_FakeTemplateCls, "build", staticmethod(hanging_build))
    with pytest.raises(concurrent.futures.TimeoutError):
        sandbox.ensure_task_template(Path("/tasks/t"), build_timeout_s=0.05)


class _FakeSandbox:
    # [CLASS DOC] HELPER: Fake e2b sandbox. Tracks whether it was killed, records every command
    # run (positional and keyword arguments) through a commands helper object, resolves a host name
    # from the port, and supports a kill call that accepts any keyword arguments.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Not killed, no commands run; builds a commands helper object whose
        # run method appends its arguments to the ran list.
        self.killed = False
        self.ran = []
        self.commands = type("C", (), {"run": lambda _self, *a, **k: self.ran.append((a, k))})()

    def get_host(self, port):
        # [FUNCTION DOC] HELPER: Returns the port followed by -id.example.test.
        return f"{port}-id.example.test"

    def kill(self, **kwargs):
        # [FUNCTION DOC] HELPER: Marks the sandbox killed (keyword arguments ignored).
        self.killed = True


def test_create_task_sandbox_kills_on_partial_failure(monkeypatch, fake_e2b):
    # [FUNCTION DOC] Installs a fake sandbox class whose create returns a fake sandbox; patches
    # template ensuring, labels and server command; makes the readiness wait raise a timeout
    # error. Asserts creating the task sandbox with a tiny ready timeout raises the timeout error
    # and that the fake sandbox was killed.
    """A sandbox whose server never becomes ready must not leak until TTL."""
    created = _FakeSandbox()
    fake_e2b.Sandbox = type("S", (), {"create": staticmethod(lambda **k: created)})
    monkeypatch.setattr(sandbox, "ensure_task_template", lambda task_dir: "tb2-t-abc")
    monkeypatch.setattr(sandbox, "sandbox_labels", lambda task_dir: {})
    monkeypatch.setattr(sandbox, "server_cmd", lambda *a, **k: "serve")

    def never_ready(url, timeout_s):
        # [FUNCTION DOC] HELPER: Always raises a timeout error.
        raise TimeoutError("server not ready")

    monkeypatch.setattr(sandbox, "wait_server_ready", never_ready)
    with pytest.raises(TimeoutError):
        sandbox.create_task_sandbox(Path("/tasks/t"), ready_timeout_s=0.01)
    assert created.killed


def test_create_runs_the_env_server_as_root(monkeypatch, fake_e2b):
    # [FUNCTION DOC] With template, labels, server command, readiness and keepalive patched out,
    # creates the task sandbox. Asserts exactly one command was run, with user root, background
    # true, and the command text containing the patched server command.
    """The env server executes the agent's commands, so the user it runs as IS
    the task environment's user. E2B defaults to a non-root user; a TB2 task
    image expects root (its own tests apt-install), so the server must be
    started as root or the agent silently loses privileges it should have."""
    created = _FakeSandbox()
    fake_e2b.Sandbox = type("S", (), {"create": staticmethod(lambda **k: created)})
    monkeypatch.setattr(sandbox, "ensure_task_template", lambda task_dir: "tb2-t-abc")
    monkeypatch.setattr(sandbox, "sandbox_labels", lambda task_dir: {})
    monkeypatch.setattr(sandbox, "server_cmd", lambda *a, **k: "serve")
    monkeypatch.setattr(sandbox, "wait_server_ready", lambda url, timeout_s: None)
    monkeypatch.setattr(sandbox, "_start_keepalive", lambda sb, task_id: None)

    sandbox.create_task_sandbox(Path("/tasks/t"))

    ((args, kwargs),) = created.ran
    assert kwargs["user"] == "root"
    assert kwargs["background"] is True
    assert "serve" in args[0]


def test_task_build_resources_from_task_toml(tmp_path: Path):
    # [FUNCTION DOC] A task config file with cpus 4 and memory 8192 megabytes in its environment
    # section yields cpu count 4 and memory 8192.
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 4\nmemory_mb = 8192\n")
    assert sandbox.task_build_resources(tmp_path) == {"cpu_count": 4, "memory_mb": 8192}


def test_task_build_resources_floors(tmp_path: Path):
    # [FUNCTION DOC] A task config with cpus 0 and memory 128 megabytes is floored to 1 cpu and 2048 megabytes.
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 0\nmemory_mb = 128\n")
    assert sandbox.task_build_resources(tmp_path) == {"cpu_count": 1, "memory_mb": 2048}


# --- key supply --------------------------------------------------------------


def test_resolve_api_key_env_value_wins(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With both the key variable and a key file variable set, the resolved key is the variable's value.
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_from_file\n")
    monkeypatch.setenv("E2B_API_KEY", "e2b_from_env")
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "e2b_from_env"


def test_resolve_api_key_falls_back_to_file(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With only the key file path variable set, the resolved key is the file's
    # content with surrounding whitespace stripped.
    # The launcher forwards only this path (never the value, which would be
    # echoed into driver logs via ray runtime_env); workers must read it here.
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_from_file\n")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    assert sandbox.resolve_api_key() == "e2b_from_file"  # whitespace stripped


def test_resolve_api_key_default_path_under_home(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With neither variable set and the home variable pointing at a temp dir whose
    # config folder holds an e2b key file, the resolved key is that file's content.
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".config" / "e2b"
    cfg.mkdir(parents=True)
    (cfg / "api_key").write_text("e2b_default\n")
    assert sandbox.resolve_api_key() == "e2b_default"


def test_resolve_api_key_errors_when_absent(monkeypatch, tmp_path: Path):
    # [FUNCTION DOC] With no key variable and a key file path naming a missing file, resolution
    # raises a runtime error saying the key is missing or empty.
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing or empty"):
        sandbox.resolve_api_key()


# --- base_url ---------------------------------------------------------------


class _HostStub:
    # [CLASS DOC] HELPER: Fake sandbox providing only a host lookup by port.
    def get_host(self, port):
        # [FUNCTION DOC] HELPER: Returns the port followed by -sbx123.example.dev.
        return f"{port}-sbx123.example.dev"


def test_base_url_default_https(monkeypatch):
    # [FUNCTION DOC] With the url scheme variable unset, the base url for the stub is https plus
    # the host for port 8000.
    monkeypatch.delenv("OPENENV_E2B_URL_SCHEME", raising=False)
    assert sandbox.base_url(_HostStub()) == "https://8000-sbx123.example.dev"


def test_base_url_scheme_override(monkeypatch):
    # [FUNCTION DOC] With the url scheme variable set to http, the base url uses http.
    # Self-hosted gateways (e.g. an AgentENV deployment inside the tailnet)
    # may terminate plain HTTP.
    monkeypatch.setenv("OPENENV_E2B_URL_SCHEME", "http")
    assert sandbox.base_url(_HostStub()) == "http://8000-sbx123.example.dev"


# --- TTL / keepalive ---------------------------------------------------------


def test_sandbox_ttl_armed_by_default():
    # [FUNCTION DOC] Asserts the module's sandbox lifetime in seconds is positive and the
    # keepalive interval is shorter than it.
    # The dead-man's-switch contract: sandboxes are created with a bounded
    # lifetime, or a hard-killed caller's orphans run forever.
    assert sandbox._SANDBOX_TTL_S > 0
    assert sandbox._KEEPALIVE_INTERVAL_S < sandbox._SANDBOX_TTL_S  # beats within the window


def test_keepalive_beats_then_exits_on_persistent_failure(monkeypatch):
    # [FUNCTION DOC] Sets the keepalive interval to 0.02 seconds and the connection options to an
    # empty mapping; starts the keepalive on a stub sandbox whose set-timeout call counts beats
    # and raises once marked dead. Waits (up to 2 seconds) for at least 3 beats, then marks it dead
    # and waits (up to 2 seconds) until no live thread has keepalive in its name; asserts none remains.
    monkeypatch.setattr(sandbox, "_KEEPALIVE_INTERVAL_S", 0.02)
    monkeypatch.setattr(sandbox, "_connection_opts", lambda: {})

    class Stub:
        # [CLASS DOC] HELPER: Fake sandbox counting timeout refreshes and failing after being marked dead.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Starts with zero beats and alive.
            self.beats = 0
            self.dead = False

        def set_timeout(self, timeout, **opts):
            # [FUNCTION DOC] HELPER: Raises a runtime error if dead; otherwise increments the beat count.
            if self.dead:
                raise RuntimeError("sandbox killed")
            self.beats += 1

    stub = Stub()
    sandbox._start_keepalive(stub, "regex-chess")
    deadline = time.time() + 2.0
    while stub.beats < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert stub.beats >= 3  # beats while the sandbox is alive

    stub.dead = True  # episode over, sandbox killed -> thread must exit
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any("keepalive" in t.name for t in threading.enumerate()):
            break
        time.sleep(0.01)
    assert not any("keepalive" in t.name for t in threading.enumerate())
