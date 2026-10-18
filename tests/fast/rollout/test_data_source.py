# [MODULE DOC] Tests for the built-in rollout data source's save and load of global dataset state.
# Checks that save and load are silent no-ops when no global dataset is used, that load logs a
# warning (on the module's logger) in each situation where nothing is restored, and that a saved
# state file restores the sample offset and epoch id. Uses real temporary directories and, in the
# last test, real tensor serialization; no GPU or network.
import logging
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.data_source import RolloutDataSource


def _make_args(**overrides) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a simple namespace of data-source args with defaults: global
    # dataset off, no save dir, no load dir, no shuffle; keyword overrides win. Returns it.
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    # [FUNCTION DOC] Builds a real data source with a save dir but the global dataset off, calls save
    # for rollout 3, and asserts the save directory is still empty.
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    # [FUNCTION DOC] Builds a real data source with a load dir but the global dataset off, calls load
    # for rollout 3, and asserts the sample offset and epoch id remain 0.
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _bare_source(**overrides) -> RolloutDataSource:
    # [FUNCTION DOC] HELPER: Creates a data source object bypassing its constructor (so no dataset is
    # loaded), setting only its args (from the make-args helper with the overrides) and an empty
    # metadata mapping. Returns it.
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = _make_args(**overrides)
    source.metadata = {}
    return source


def test_load_says_so_when_it_finds_no_state(tmp_path: Path, caplog) -> None:
    # [FUNCTION DOC] With the global dataset on and a load dir containing no state file, load for
    # rollout 3 must log a warning containing the text "no dataset state under".
    """A dataset silently starting over is a run replaying samples its trainers already trained on."""
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no dataset state under" in caplog.text


def test_load_says_so_when_the_run_names_no_load_directory(tmp_path: Path, caplog) -> None:
    # [FUNCTION DOC] With the global dataset on but no load dir, load must log a warning containing
    # "no --load". The temp path argument is unused.
    """A run told to write but not to read still has to say that its dataset starts from the beginning."""
    source = _bare_source(rollout_global_dataset=True, load=None)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no --load" in caplog.text


def test_load_says_so_when_the_run_keeps_no_global_dataset(tmp_path: Path, caplog) -> None:
    # [FUNCTION DOC] With the global dataset off but a load dir given, load must log a warning
    # mentioning the rollout-global-dataset option.
    """A custom rollout function keeps its own state, and the operator has to know this one restored none."""
    source = _bare_source(rollout_global_dataset=False, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "rollout-global-dataset" in caplog.text


def test_load_restores_the_state_it_finds(tmp_path: Path) -> None:
    # [FUNCTION DOC] Writes a state file at the path computed for the temp dir and rollout 3,
    # containing sample offset 7 and epoch id 1 (saved with the tensor library, parent directories
    # created), then loads into a bare source with the global dataset on, and asserts the source now
    # reports offset 7 and epoch 1.
    """This is the ordinary resume, and the position it restores is what keeps a run off samples it has seen."""
    import torch

    from miles.rollout.data_source import compute_global_dataset_state_path

    path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_offset": 7, "epoch_id": 1}, path)
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    source.load(rollout_id=3)

    assert (source.sample_offset, source.epoch_id) == (7, 1)
