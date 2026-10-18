# [MODULE DOC] Unit tests for the wandb utilities: primary and secondary initialisation (the
# wandb library calls are replaced with recorders; the init uses shared mode, a 300 second init
# timeout, and resumes a pre-assigned run id), and the config computed for logging (the
# launcher's environment report is embedded when its file can be read, and omitted otherwise).
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fast.utils.env_report.conftest import make_args

from miles.utils.tracking_utils import wandb_utils
from miles.utils.tracking_utils.wandb_utils import _compute_config_for_logging


class TestPrimaryWandbInit:
    # [CLASS DOC] Test of the primary initialisation resuming an assigned run.
    def test_primary_wandb_init_resumes_a_preassigned_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Builds shared test args (rank 0, wandb on, run id assigned) and patches
        # wandb init (recording keyword arguments), define metric (no-op) and the run object (with
        # the assigned id). After the primary init, the first init call has the assigned id and the
        # resume mode allow.
        """A replacement primary resumes the explicitly assigned WandB run."""
        init_calls: list[dict[str, object]] = []
        args = make_args(
            rank=0,
            use_wandb=True,
            wandb_dir=None,
            wandb_group="group",
            wandb_host=None,
            wandb_key=None,
            wandb_mode=None,
            wandb_project="project",
            wandb_random_suffix=False,
            wandb_run_id="assigned-run",
            wandb_team=None,
        )
        monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
        monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
        monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="assigned-run"), raising=False)

        wandb_utils.init_wandb_primary(args)

        assert init_calls[0]["id"] == "assigned-run"
        assert init_calls[0]["resume"] == "allow"


class TestTheLoggedConfig:
    # [CLASS DOC] Tests of the logged config's launcher environment report entry.
    def test_carries_the_record_of_the_launch_that_started_this_run(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Writes a launcher record JSON file and points the args at it; the config
        # has a launcher env report entry with the run id and worker argument vector from the file.
        """The launcher's record is what an algorithm engineer looks for first, and wandb is where they look."""
        record = tmp_path / "launch-1.json"
        record.write_text(json.dumps({"run_id": "260101-000000-000", "worker_argv": ["--lr", "1"]}))

        config = _compute_config_for_logging(make_args(env_report=str(record)))

        assert config["launcher_env_report"]["run_id"] == "260101-000000-000"
        assert config["launcher_env_report"]["worker_argv"] == ["--lr", "1"]

    def test_names_no_launcher_record_when_no_launcher_wrote_one(self) -> None:
        # [FUNCTION DOC] With default args (no report path), the config has no launcher env
        # report key.
        """A run started by hand has no record, and wandb must not grow an empty key for it."""
        assert "launcher_env_report" not in _compute_config_for_logging(make_args())

    def test_survives_a_record_the_shared_disk_lost(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A report path to a nonexistent file does not raise and the config has no
        # launcher env report key.
        """Losing the record must cost the run its audit trail, not its tracking."""
        config = _compute_config_for_logging(make_args(env_report=str(tmp_path / "gone.json")))

        assert "launcher_env_report" not in config


def _args(**overrides):
    # [FUNCTION DOC] HELPER: Builds a plain namespace with wandb defaults (no env report, rank 0,
    # metrics off, wandb on, group group, project project, no random suffix, no run id or other
    # optional settings) updated by the overrides.
    values = {
        "env_report": None,
        "rank": 0,
        "sglang_enable_metrics": False,
        "use_wandb": True,
        "wandb_dir": None,
        "wandb_group": "group",
        "wandb_host": None,
        "wandb_key": None,
        "wandb_mode": None,
        "wandb_project": "project",
        "wandb_random_suffix": False,
        "wandb_run_id": None,
        "wandb_team": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_primary_wandb_init_uses_extended_init_timeout(monkeypatch):
    # [FUNCTION DOC] With the wandb library patched, primary init passes settings with shared
    # mode, the primary flag true and an init timeout of 300 seconds, and stores the run id on
    # the args (run-id).
    init_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="run-id"), raising=False)

    args = _args()
    wandb_utils.init_wandb_primary(args)

    settings = init_calls[0]["settings"]
    assert settings.mode == "shared"
    assert settings.x_primary is True
    assert settings.init_timeout == 300.0
    assert args.wandb_run_id == "run-id"


def test_primary_wandb_init_resumes_a_preassigned_run(monkeypatch):
    # [FUNCTION DOC] Module level variant of the class test above using plain args: the first
    # init call has the assigned id and resume mode allow. (Duplicates the class based test with
    # a different args builder.)
    """An explicitly assigned run ID lets a replacement primary resume the run."""
    init_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="run-id"), raising=False)

    wandb_utils.init_wandb_primary(_args(wandb_run_id="assigned-run"))

    assert init_calls[0]["id"] == "assigned-run"
    assert init_calls[0]["resume"] == "allow"


def test_secondary_wandb_init_uses_extended_init_timeout(monkeypatch):
    # [FUNCTION DOC] With the wandb library patched, secondary init for run id run-id passes
    # settings with shared mode, the primary flag false, finish-state updates disabled, and an
    # init timeout of 300 seconds.
    init_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)

    args = _args(wandb_run_id="run-id")
    wandb_utils.init_wandb_secondary(args)

    settings = init_calls[0]["settings"]
    assert settings.mode == "shared"
    assert settings.x_primary is False
    assert settings.x_update_finish_state is False
    assert settings.init_timeout == 300.0
