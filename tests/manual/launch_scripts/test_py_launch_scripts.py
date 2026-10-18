# [MODULE DOC] Snapshot tests for every Python launcher script (scripts/**/run_*.py style).
# Each launcher entrypoint is imported with a frozen environment (cleared env knobs, pinned
# hardware, hidden host filesystem), called with a command recorder installed in place of real
# shell execution, and the recorded shell commands are compared with a golden text file under
# tests/snapshots/launch_scripts/py/<script>/<entrypoint>.txt. A second group of tests checks
# the harness itself (filesystem freezing) and that the discovery lists, hardware pins and
# golden files stay consistent with each other.
# The scripts listed in the first denylist need extra inputs (fake checkpoints or model names)
# because their defaults are unsupported; the NPU launcher is excluded because it cannot be
# imported in this checkout.
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    CLEARED_ENV,
    FROZEN_HARDWARE,
    call_entrypoint,
    format_recording,
    freeze_environment,
    host_filesystem_frozen,
    import_launch_script,
    install_command_recorder,
    iter_py_launch_scripts,
    launcher_hardware_literals,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "py"

_SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH = {"scripts/run_qwen3_4b_npu.py"}


def _glm_checkpoint(sandbox: Path, model_name: str, num_layers: int) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Creates a fake GLM MoE-DSA checkpoint directory in the sandbox.
    # Makes a models directory under the sandbox with a subdirectory named after the model that
    # holds only a config file declaring the glm_moe_dsa model type, its causal-LM architecture
    # and the given number of hidden layers. Returns the keyword arguments for the entrypoint:
    # a mapping with the model directory path as a string. Side effect: writes to the sandbox;
    # raises if the directory already exists.
    model_dir = sandbox / "models"
    (model_dir / model_name).mkdir(parents=True)
    (model_dir / model_name / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmMoeDsaForCausalLM"],
                "num_hidden_layers": num_layers,
            }
        )
    )
    return {"model_dir": str(model_dir)}


def _nemotron_checkpoint(sandbox: Path) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Creates a fake Nemotron-3 Nano 4B checkpoint directory in the sandbox.
    # Writes a config file (model type nemotron_h, with an auto-config mapping to the Nemotron-H
    # configuration class) into a checkpoint directory named after the BF16 Nano 4B model under
    # a models directory in the sandbox. Returns a mapping with the model directory path as a
    # string, used as entrypoint keyword arguments. Raises if the directory already exists.
    model_dir = sandbox / "models"
    checkpoint = model_dir / "NVIDIA-Nemotron-3-Nano-4B-BF16"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "auto_map": {"AutoConfig": "configuration_nemotron_h.NemotronHConfig"},
            }
        )
    )
    return {"model_dir": str(model_dir)}


_SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED: dict[str, Callable[[Path], dict[str, object]]] = {
    "scripts/run_deepseek_v4.py": lambda sandbox: {"model_name": "DeepSeek-V4-Flash-FP8-4layer"},
    "scripts/run_glm5_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5", 78),
    "scripts/run_glm5_2_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5.2", 78),
    "scripts/run_inkling.py": lambda sandbox: {"model_name": "Inkling-4layer"},
    # convert_checkpoint locks a file under model_dir, which must be writable while recording
    "scripts/run_kimi_k3.py": lambda sandbox: {"model_dir": str(sandbox / "models")},
    # test_prepare_waits_for_shared_checkpoint_conversion covers the real cache lock.
    "scripts/run_mimo_v2_6_flash.py": lambda sandbox: {"model_dir": str(sandbox / "models")},
    "scripts/run_nemotron_3_nano_4b_fsdp.py": _nemotron_checkpoint,
    "scripts/run_nemotron_3_ultra_550b_a55b.py": lambda sandbox: {
        "model_name": "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer"
    },
}

# The machine each recording represents. Launchers default --hardware to whatever node they run on, so the
# suite pins one; FROZEN_HARDWARE covers the rest.
_HARDWARE_A_RECORDING_REPRESENTS = {
    "scripts/amd/run_gpt_oss_20b_lora.py": "MI355X",
    "scripts/amd/run_qwen3_30b_a3b.py": "MI355X",
    "scripts/amd/run_qwen3_4b.py": "MI355X",
    "scripts/amd/run_qwen3_4b_lora.py": "MI355X",
    "scripts/run_deepseek_v32.py": "B200",
    "scripts/run_glm45_355b_a32b.py": "GB200",
    "scripts/run_joy_ai_llm_flash.py": "B200",
    "scripts/run_mcore_fsdp.py": "H100",
    "scripts/run_qwen3_30b_a3b.py": "H100",
    "scripts/run_qwen3_4b.py": "H100",
}

_ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS = {
    ("scripts/run_deepseek_v4.py", "prepare_mxfp8"),
    ("scripts/run_deepseek_v4.py", "prepare_fp8"),
}

_SCRIPTS = [
    script for script in iter_py_launch_scripts() if script.rel not in _SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH
]
_CASES = [(script.rel, entrypoint) for script in _SCRIPTS for entrypoint in script.entrypoints]


@pytest.fixture(params=_CASES, ids=[f"{rel}::{entrypoint}" for rel, entrypoint in _CASES])
def recorded(request, monkeypatch, tmp_path):
    # [FUNCTION DOC] HELPER: Parametrized fixture over every (script path, entrypoint name) case.
    # Freezes the environment with the hardware pinned for that script (or the default frozen
    # hardware), installs the command recorder, imports the launcher script from the repo root,
    # and calls the entrypoint with the per-script extra arguments (empty if the script has no
    # entry in the unsupported-defaults table), using the temporary directory as sandbox.
    # Returns a tuple of script path, entrypoint name, the recording, and the sandbox path.
    rel, entrypoint = request.param
    freeze_environment(monkeypatch, hardware=_HARDWARE_A_RECORDING_REPRESENTS.get(rel, FROZEN_HARDWARE))
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / rel)
    call_entrypoint(
        module,
        entrypoint,
        _SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED.get(rel, lambda sandbox: {})(tmp_path),
        sandbox=tmp_path,
    )
    return rel, entrypoint, recording, tmp_path


class TestEveryLauncherEntrypoint:
    # [CLASS DOC] Per-case checks run once for every launcher entrypoint produced by the fixture.
    def test_commands_match_snapshot(self, recorded):
        # [FUNCTION DOC] Compares the recorded commands, formatted relative to the sandbox, with
        # the golden file named after the script and entrypoint; the snapshot helper reports a
        # mismatch (or creates/updates per its own rules).
        """Every launcher entrypoint must build exactly the recorded shell commands."""
        rel, entrypoint, recording, sandbox = recorded
        snapshot = _SNAPSHOT_DIR / rel / f"{entrypoint}.txt"

        assert_matches_snapshot(snapshot, format_recording(recording, sandbox=sandbox), f"{rel}::{entrypoint}")

    def test_entrypoint_issues_commands(self, recorded):
        # [FUNCTION DOC] Asserts the recording is empty for the entrypoints whose own defaults
        # disable them (the DeepSeek-V4 mxfp8 and fp8 prepare steps), and non-empty for all others.
        """An entrypoint that silently does nothing is a broken launcher, not a passing test."""
        rel, entrypoint, recording, _ = recorded
        if (rel, entrypoint) in _ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS:
            assert not recording.commands
        else:
            assert recording.commands


class TestHostFilesystemIsFrozen:
    # [CLASS DOC] Tests of the harness context manager that hides the host filesystem from launchers.
    def test_paths_outside_the_checkout_and_the_sandbox_report_absence(self, tmp_path):
        # [FUNCTION DOC] Inside the freeze, a file written in the sandbox still exists while a
        # host path under the root home directory (a checkpoint index file) reports as absent.
        """A launcher that can see the host's checkpoints skips work, so the snapshot would follow the machine."""
        inside = tmp_path / "checkpoint.json"
        inside.write_text("{}")

        with host_filesystem_frozen(tmp_path):
            assert inside.exists()
            assert not Path("/root/models/some-checkpoint/model.safetensors.index.json").exists()

    def test_the_checkout_stays_visible(self, tmp_path):
        # [FUNCTION DOC] Inside the freeze, the repository's project file and its scripts/models
        # directory must still be reported as existing.
        """A launcher resolves its own model args script out of the checkout, so hiding it breaks every entrypoint."""
        with host_filesystem_frozen(tmp_path):
            assert (REPO_ROOT / "pyproject.toml").exists()
            assert (REPO_ROOT / "scripts" / "models").exists()

    def test_an_unreadable_parent_reports_absence_instead_of_raising(self, tmp_path):
        # [FUNCTION DOC] Creates a directory with all permissions removed, then inside the freeze
        # checks that asking whether a file beneath it exists returns false instead of raising a
        # permission error. Permissions are restored in a cleanup block so the temp dir can be
        # deleted. (Note: permission bits do not block the root user, so this may be vacuous there.)
        """python 3.11 raises PermissionError from exists(), which is how the CPU runner's /root broke this."""
        unreadable = tmp_path / "unreadable"
        unreadable.mkdir()
        unreadable.chmod(0o000)
        try:
            with host_filesystem_frozen(tmp_path / "sandbox"):
                assert not (unreadable / "model.safetensors.index.json").exists()
        finally:
            unreadable.chmod(0o700)


class TestDiscovery:
    # [CLASS DOC] Consistency checks between discovered launchers, pins, goldens and frozen env.
    def test_all_py_launch_scripts_are_discovered(self):
        # [FUNCTION DOC] Asserts more than fifteen launcher scripts are covered, so an empty
        # discovery result cannot pass silently.
        """Guards against the discovery glob silently going empty."""
        assert len(_SCRIPTS) > 15

    def test_every_discovered_launcher_is_covered_except_the_one_this_checkout_cannot_import(self):
        # [FUNCTION DOC] The set difference between all discovered launchers and the covered
        # ones must equal exactly the NPU-only exclusion set.
        """A denylist that nobody rechecks only grows; name the survivors so the count cannot drift."""
        discovered = {script.rel for script in iter_py_launch_scripts()}

        assert discovered - {script.rel for script in _SCRIPTS} == _SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH

    @pytest.mark.parametrize("rel", sorted(_SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH))
    def test_the_uncovered_launcher_really_is_uncoverable_here(self, rel):
        # [FUNCTION DOC] Parametrized over the excluded launchers: importing each must raise an
        # import error whose message mentions the missing NPU training module, so the exclusion
        # is removed once that dependency is available.
        """Once the NPU patch is upstreamed this fails, forcing the exclusion out instead of letting it rot."""
        with pytest.raises(ImportError, match="execute_train_npu"):
            import_launch_script(REPO_ROOT / rel)

    def test_every_pinned_recording_names_a_launcher_that_accepts_that_hardware(self):
        # [FUNCTION DOC] Takes the mapping from launcher path to hardware literals the launcher
        # accepts; asserts every pinned script key exists in it and each pinned hardware name is
        # among that launcher's accepted literals.
        """A pin the launcher does not accept records a profile nobody can run, and a stale key pins nothing."""
        literals = launcher_hardware_literals()

        assert _HARDWARE_A_RECORDING_REPRESENTS.keys() <= literals.keys()
        for rel, hardware in _HARDWARE_A_RECORDING_REPRESENTS.items():
            assert hardware in literals[rel], rel

    def test_a_launcher_the_frozen_default_covers_is_not_pinned_as_well(self):
        # [FUNCTION DOC] Asserts no pin equals the default frozen hardware (such a pin is redundant).
        """Two ways to say the same thing drift apart; only the launchers FROZEN_HARDWARE cannot serve get a pin."""
        redundant = {rel for rel, hardware in _HARDWARE_A_RECORDING_REPRESENTS.items() if hardware == FROZEN_HARDWARE}

        assert not redundant

    def test_every_discovered_entrypoint_has_a_golden_and_vice_versa(self):
        # [FUNCTION DOC] The set of expected golden paths (script/entrypoint.txt) built from the
        # cases must equal the set of text files found recursively under the snapshot directory.
        """A removed entrypoint leaves its golden behind, and a new one would otherwise go unrecorded."""
        expected = {f"{rel}/{entrypoint}.txt" for rel, entrypoint in _CASES}
        recorded = {path.relative_to(_SNAPSHOT_DIR).as_posix() for path in _SNAPSHOT_DIR.rglob("*.txt")}

        assert expected == recorded

    def test_the_snapshot_tree_holds_nothing_outside_the_three_recorded_families(self):
        # [FUNCTION DOC] The parent of the py snapshot directory must contain exactly the three
        # family directories: py, sh and self_executing.
        """A whole orphan directory would sit under tests/snapshots/launch_scripts with nobody checking it."""
        root = _SNAPSHOT_DIR.parent
        families = {root / name for name in ("py", "sh", "self_executing")}

        assert set(root.iterdir()) == families

    def test_every_environment_knob_a_model_script_reads_is_frozen(self):
        # [FUNCTION DOC] Scans each regular file directly in scripts/models, collecting upper-case
        # variable names used with a shell default-value expansion and names passed to the Python
        # environment getter with a string literal. Asserts the collected set is non-empty and a
        # subset of the harness's list of cleared environment variables.
        """The snapshots now pin expanded model args, so a developer's exported override would fail them."""
        knobs = set()
        for script in sorted((REPO_ROOT / "scripts" / "models").iterdir()):
            if not script.is_file():
                continue
            text = script.read_text()
            knobs |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):-", text))
            knobs |= set(re.findall(r"environ\.get\(\s*\"([A-Z][A-Z0-9_]*)\"", text))

        assert knobs
        assert knobs <= set(CLEARED_ENV)

    def test_execute_train_config_defaults_are_not_taken_from_a_slurm_allocation(self, monkeypatch):
        # [FUNCTION DOC] Imports the command utilities module and asserts a default-constructed
        # training config has one node.
        # KNOWN ISSUE: the monkeypatch fixture is unused, so no SLURM variable is cleared or set
        # by this test itself; it relies on module import-time state being already frozen.
        """SLURM_JOB_NUM_NODES is read at import time, so a stale allocation would skew every snapshot."""
        import miles.utils.external_utils.command_utils as command_utils

        assert command_utils.ExecuteTrainConfig().num_nodes == 1
