# [MODULE DOC]
# Tests for the per-task sandbox recipe module of the openenv terminal-bench-2 example (imported
# by bare name as recipe). A helper creates a minimal git checkout of a tasks repository in a temp
# dir (a git repository with a GitHub-style origin url, one task folder with a task config naming a
# docker image, committed with a throwaway identity; this runs real git subprocesses). Tests
# cover: the task layer command excluding exactly that task's solution folder and ending with the
# task path inside a commit-named archive prefix; deterministic base64 tar output of directories
# (no gzip header timestamps or file mtime leaking in); the server command setting the withhold
# tests gate and the default task id; and the floors applied to the task config's resource values.
"""Tests for the per-task sandbox recipe's verifier-asset hygiene.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q
"""

import os
import subprocess
import time
from pathlib import Path

import tb2_sandbox_recipe as recipe


def _make_tasks_repo(root: Path, task_name: str = "some-task") -> Path:
    # [FUNCTION DOC] HELPER: Creates a tb2repo folder under root with a task folder of the given
    # name (default some-task) containing a task config file whose environment section sets the
    # docker image debian:12. Runs git init, adds an origin remote at a GitHub url for acme/tb2tasks,
    # stages everything and commits with a throwaway identity (each a real subprocess in the repo,
    # raising on failure). Returns the task folder path.
    """A minimal tasks checkout: git repo with a GitHub origin and one task."""
    repo = root / "tb2repo"
    task = repo / task_name
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[environment]\ndocker_image = "debian:12"\n')
    for cmd in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", "https://github.com/acme/tb2tasks.git"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@e.st", "-c", "user.name=t", "commit", "-qm", "x"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return task


def test_task_layer_excludes_solution(tmp_path: Path):
    # [FUNCTION DOC] Builds the repo, reads its head commit hash via git, and computes the task
    # layer command. Asserts the command excludes the task's solution folder and its contents
    # (patterns prefixed by tb2tasks-<commit>/some-task/solution), and, ignoring trailing
    # whitespace, ends with the quoted tb2tasks-<commit>/some-task path.
    task = _make_tasks_repo(tmp_path)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=task.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    cmd = recipe._task_layer_command(task)

    # The exclusion is anchored to this one task's solution/ (dir + contents),
    # not a loose pattern that could drop task files elsewhere.
    assert f"--exclude='tb2tasks-{sha}/some-task/solution'" in cmd
    assert f"--exclude='tb2tasks-{sha}/some-task/solution/*'" in cmd
    assert cmd.rstrip().endswith(f"'tb2tasks-{sha}/some-task'")


def test_dir_tar_b64_is_deterministic(tmp_path: Path):
    # [FUNCTION DOC] Creates a small package folder with a file and a subfolder file, encodes it
    # once, sleeps 1.1 seconds (to cross a gzip header timestamp second), sets a file's mtime to
    # the epoch, encodes again with the same arguments (one source dir, one archive name, a size
    # limit of 100000 bytes), and asserts the two outputs are identical. Side effects: files in
    # the temp path, a real sleep.
    # The b64 is embedded in a build command: identical source must yield
    # identical bytes, or every call produces a new image definition and
    # provider build caches / pre-baked snapshots never hit. Guards the two
    # nondeterminism sources: the gzip header timestamp (differs between
    # calls in different seconds) and per-entry mtime/owner metadata
    # (differs between checkouts of the same content).
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    sub = src / "server"
    sub.mkdir()
    (sub / "b.py").write_text("y = 2\n")

    first = recipe._dir_tar_b64([src], ["pkg"], max_bytes=100_000)
    time.sleep(1.1)  # cross a gzip-header timestamp boundary
    os.utime(src / "a.py", (0, 0))  # same content, different file mtime
    second = recipe._dir_tar_b64([src], ["pkg"], max_bytes=100_000)

    assert first == second


def test_server_cmd_sets_withhold_gate():
    # [FUNCTION DOC] Asserts the default server command contains the setting TB2_WITHHOLD_TESTS=1.
    assert "TB2_WITHHOLD_TESTS=1" in recipe.server_cmd()


def test_server_cmd_defaults_to_the_staged_task():
    # [FUNCTION DOC] Asserts a server command built with default task id fix-git contains
    # TB2_DEFAULT_TASK_ID=fix-git followed by a space.
    # A per-task sandbox stages one task; a reset() with no task_id must land
    # on it, not the env's built-in headless-terminal default.
    cmd = recipe.server_cmd(default_task_id="fix-git")
    assert "TB2_DEFAULT_TASK_ID=fix-git " in cmd


def test_task_env_resources_floors(tmp_path: Path):
    # [FUNCTION DOC] A task config with cpus 0, memory 128 and storage 1024 yields the floored
    # triple (1 cpu, 2048 memory, 10240 storage); a config with cpus 4, memory 8192 and storage
    # 20480 yields those values unchanged.
    """The floors are recipe policy, proven once here; the per-provider tests
    pin only each materialization's unit/kwarg mapping."""
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 0\nmemory_mb = 128\nstorage_mb = 1024\n")
    assert recipe.task_env_resources(tmp_path) == (1, 2048, 10240)

    (tmp_path / "task.toml").write_text("[environment]\ncpus = 4\nmemory_mb = 8192\nstorage_mb = 20480\n")
    assert recipe.task_env_resources(tmp_path) == (4, 8192, 20480)
