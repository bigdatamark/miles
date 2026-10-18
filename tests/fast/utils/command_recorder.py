# [MODULE DOC] HELPER: Test helper that intercepts command execution of the external command
# utilities. Provides a function to patch a method on the base command backend class and a function
# that replaces every command-execution entry point (CPU, GPU and multi-node execution on the
# backend, the CPU command function and the launcher-owned Ray job runner of the Ray job module)
# with recorders that append a text form of each command to a list and never run anything.
import json
import shlex

import miles.utils.external_utils.ray_job as ray_job
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend


def patch_helper(monkeypatch, name: str, replacement, *, backend_class: type = BaseCommandBackend) -> None:
    # [FUNCTION DOC] HELPER: Replace a named attribute of a backend class (default the base command
    # backend) using the monkeypatch fixture. Asserts first that the class already has an attribute
    # of that name (failing with a message naming class and attribute); the replacement is installed
    # with the non-raising option, undone at test teardown.
    assert hasattr(backend_class, name), f"no method of {backend_class.__name__} is named {name}"
    monkeypatch.setattr(backend_class, name, replacement, raising=False)


def record_commands(monkeypatch) -> list[str]:
    # [FUNCTION DOC] HELPER: Install recorders and return the list they append to (initially
    # empty). Patched: the Ray job module's CPU command function (forwarded to the single-command
    # recorder with no instance), its launcher-owned job runner, and the backend class's CPU, GPU
    # and multi-node command methods. Each recorded entry is a string: the command itself; for
    # multi-node "[multi_node num_nodes=N] " prefix plus command; for the launcher runner a
    # "[launcher lifetime] ray job submit" line with shell-quoted address and runtime-environment JSON
    # followed by the entrypoint. Nothing is executed. All patches are undone at teardown.
    """Replace every exec_command backend method with a recorder and return the list it appends to.

    Patching BaseCommandBackend covers every backend, including one added after this was written: the
    public forms are defined there and no subclass is allowed to hide them behind an override.
    """
    commands: list[str] = []

    def fake_exec_command(self, cmd: str, capture_output: bool = False, **kwargs) -> str | None:
        # [FUNCTION DOC] HELPER: Record the command string; return the string "0" when output
        # capture was requested, else none. Extra keyword arguments are ignored.
        commands.append(cmd)
        return "0" if capture_output else None

    def fake_exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        # [FUNCTION DOC] HELPER: Record "[multi_node num_nodes=N] " plus the command and return the
        # list ["0"] regardless of the node count. The capture flag and GPUs per node are ignored.
        commands.append(f"[multi_node num_nodes={num_nodes}] {cmd}")
        return ["0"]

    def fake_run_launcher_owned_job(*, address, entrypoint, runtime_env):
        # [FUNCTION DOC] HELPER: Record a single line describing a launcher-lifetime Ray job
        # submission: the shell-quoted address, the shell-quoted JSON of the runtime environment, and
        # the entrypoint text. Returns nothing; submits nothing.
        commands.append(
            f"[launcher lifetime] ray job submit --address={shlex.quote(address)} "
            f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} -- {entrypoint}"
        )

    monkeypatch.setattr(ray_job, "exec_command_cpu", lambda cmd, **kwargs: fake_exec_command(None, cmd, **kwargs))
    monkeypatch.setattr(ray_job, "_run_launcher_owned_job", fake_run_launcher_owned_job)
    patch_helper(monkeypatch, "exec_command_cpu", fake_exec_command)
    patch_helper(monkeypatch, "exec_command_gpu", fake_exec_command)
    patch_helper(monkeypatch, "exec_command_multi_node", fake_exec_command_multi_node)

    return commands
