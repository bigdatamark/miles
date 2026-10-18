# [MODULE DOC]
# Tests of the MiMo V2.6 Flash launcher script's prepare and execute steps using the python launch
# harness (frozen environment and command recorder): which download, conversion and training commands
# are issued per mode and checkpoint presence, and that preparation waits on a shared filesystem lock
# for a concurrent checkpoint conversion.
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.external_utils.command_utils import exclusive_path_lock
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend


@pytest.mark.parametrize("mode", ["rl", "sft"])
@pytest.mark.parametrize("checkpoint_exists", [False, True])
def test_prepare_and_execute_with_the_configured_backend(monkeypatch, tmp_path, mode, checkpoint_exists):
    # [FUNCTION DOC] Parametrized over mode (rl, sft) and whether a converted checkpoint exists. Builds
    # script args with model and data dirs in the temp dir; if the checkpoint should exist, creates its
    # directory with a safetensors index file. Runs prepare then execute and asserts over the recorded
    # commands: the model download and the bf16 conversion script run exactly when the checkpoint is
    # absent; the math dataset download runs exactly in rl mode; exactly one ray job submit command is
    # issued, and it runs the synchronous train script in rl mode and the async train script in sft mode.
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash.py")
    args = launcher.ScriptArgs(mode=mode, model_dir=str(tmp_path / "models"), data_dir=str(tmp_path / "data"))
    if checkpoint_exists:
        checkpoint = tmp_path / "models" / args.model_name
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors.index.json").write_text("{}")

    launcher.prepare(args)
    launcher.execute(args)

    commands = recording.commands
    assert any("hf download XiaomiMiMo/MiMo-V2.6-Flash-RL" in cmd for cmd in commands) is not checkpoint_exists
    assert any("convert_mimo_v2_to_bf16.py" in cmd for cmd in commands) is not checkpoint_exists
    assert any("hf download --repo-type dataset zhuzilin/dapo-math-17k" in cmd for cmd in commands) is (mode == "rl")
    train_commands = [cmd for cmd in commands if "ray job submit" in cmd]
    assert len(train_commands) == 1
    train_script = "train.py" if mode == "rl" else "train_async.py"
    assert f"/{train_script} " in train_commands[0]


def test_prepare_waits_for_shared_checkpoint_conversion(monkeypatch, tmp_path):
    # [FUNCTION DOC] Replaces the backend's gpu command execution with a function that sets a converted
    # event. Holds the exclusive path lock on the checkpoint path in the main thread, runs prepare in a
    # one-worker thread, waits (5 s) for it to start, then waits 0.2 s to see whether conversion was
    # attempted; then creates the checkpoint with its index file and releases the lock, and waits (5 s)
    # for prepare to finish. Asserts conversion was not attempted while locked, and never at all (since
    # the checkpoint appeared once the lock was released).
    freeze_environment(monkeypatch)
    install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash.py")
    args = launcher.ScriptArgs(mode="sft", model_dir=str(tmp_path / "models"))
    target = tmp_path / "models" / args.model_name
    started, converted = Event(), Event()
    monkeypatch.setattr(BaseCommandBackend, "exec_command_gpu", lambda *args, **kwargs: converted.set())

    def prepare():
        # [FUNCTION DOC] HELPER: Signals the started event, then runs the launcher's prepare step.
        started.set()
        launcher.prepare(args)

    # Preparation takes a real filesystem lock even when its shell commands are
    # recorded, so the launcher snapshots also need a writable model_dir.
    with ThreadPoolExecutor(max_workers=1) as pool:
        with exclusive_path_lock(str(target)):
            future = pool.submit(prepare)
            assert started.wait(timeout=5)
            attempted_conversion = converted.wait(timeout=0.2)
            target.mkdir(parents=True)
            (target / "model.safetensors.index.json").write_text("{}")
        future.result(timeout=5)

    assert not attempted_conversion
    assert not converted.is_set()
