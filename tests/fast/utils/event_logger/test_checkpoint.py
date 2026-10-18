# [MODULE DOC] Unit tests for the event logger checkpoint integration. Three operations act on the
# live event directory (set by the debug event data argument) and a checkpoint directory:
# snapshot copies the live events into a per-iteration folder (named iter_ plus a zero-padded
# seven digit iteration, subfolder debug_events) under the save directory; restore, when resuming
# from a load directory whose tracker file names a numeric iteration with a snapshot, replaces the
# live directory by that snapshot; discard moves the live directory's content aside into a hidden
# trash folder next to it and leaves an empty directory. Tests cover round trips, named trainers
# (checkpoints under a per-trainer namespace of the save directory), and all no-op or refusal
# cases. Everything runs in temporary directories.
"""Tests for miles.utils.audit_utils.event_logger.checkpoint."""

from argparse import Namespace
from pathlib import Path

import pytest

from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint


def _args(
    *,
    event_dir: Path | None,
    save: Path | None = None,
    load: Path | None = None,
    requested_load: Path | None = None,
    megatron_config: str | None = None,
) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the argument namespace the checkpoint functions read: the
    # debug event data directory, save directory, load directory (each as a string, or none when
    # not given), the requested load directory (defaults to the load directory when not given,
    # none when neither is set) and the encoded megatron config string (default none).
    return Namespace(
        save_debug_event_data=str(event_dir) if event_dir else None,
        save=str(save) if save else None,
        load=str(load) if load else None,
        requested_load=str(one) if (one := requested_load or load) else None,
        megatron_config=megatron_config,
    )


def _write_tracker(ckpt: Path, content: str) -> None:
    # [FUNCTION DOC] HELPER: Creates the checkpoint directory (with parents) and writes the
    # tracker file "latest_checkpointed_iteration.txt" in it with the given text content.
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "latest_checkpointed_iteration.txt").write_text(content)


class TestSnapshotRestoreRoundtrip:
    # [CLASS DOC] Tests that restore returns the live directory to the content captured by
    # snapshot, and that snapshots for one iteration overwrite each other.
    def test_restore_replaces_live_dir_with_snapshot(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Snapshots an events directory containing one file at iteration 3,
        # then changes that file and adds a second file, writes a tracker saying 3 and restores.
        # Asserts the first file has its snapshot content again and the extra file is gone.
        """A resumed run sees exactly the snapshotted events, not the live dir's leftovers."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("committed\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=3)

        # Events written after the save (would be re-executed by the resumed run).
        (events / "main.jsonl").write_text("committed\nrewound-future\n")
        (events / "straggler.jsonl").write_text("late\n")
        _write_tracker(ckpt, "3")
        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "committed\n"
        assert not (events / "straggler.jsonl").exists()

    def test_a_named_trainer_restores_from_the_tracker_in_its_own_namespace(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With an encoded megatron config for a named trainer ("policy"), the
        # snapshot is taken at iteration 3 and the tracker is written under the trainers folder
        # of the checkpoint directory in the subfolder "policy-actor" (the trainer id) instead of
        # at the root. Restore must find that tracker and bring back the snapshotted content.
        """Such a run writes every trainer's checkpoints, tracker included, under `<save>/trainers/<trainer_id>/`."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("committed\n")
        config = encode_megatron_config("policy")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt, megatron_config=config), iteration=3)

        (events / "main.jsonl").write_text("committed\nrewound-future\n")
        _write_tracker(ckpt / "trainers" / "policy-actor", "3")
        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt, megatron_config=config))

        assert (events / "main.jsonl").read_text() == "committed\n"

    def test_snapshot_overwrites_previous_snapshot_of_same_iteration(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Snapshots iteration 1 twice with different file content ("v1" then
        # "v2") and asserts the stored snapshot file holds "v2".
        """Re-saving the same iteration replaces its snapshot."""
        ckpt = tmp_path / "ckpt"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("v1\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=1)
        (events / "main.jsonl").write_text("v2\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ckpt), iteration=1)

        assert (ckpt / "iter_0000001" / "debug_events" / "main.jsonl").read_text() == "v2\n"


class TestNoOpCases:
    # [CLASS DOC] Tests of situations in which snapshot or restore must leave everything
    # untouched.
    def test_restore_skips_when_not_resuming(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Without a load directory, restore leaves the live file as it was.
        """No --load means no restore."""
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_skips_when_checkpoint_has_no_snapshot(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A load directory with a tracker for iteration 2 but no snapshot folder
        # leaves the live file unchanged.
        """Checkpoints predating event snapshots leave the live dir untouched."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "2")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_skips_a_reference_checkpoint_a_fresh_run_fell_back_to(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A reference directory holds a snapshot and tracker for iteration 7. The
        # run asked to load from another directory ("run", the requested load) but fell back to
        # the reference (the effective load). Restore must not import the reference's events:
        # the live file keeps "fresh".
        """A run whose --load holds nothing resumable must not import the reference events."""
        ref = tmp_path / "ref"
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("fresh\n")
        event_logger_checkpoint.snapshot(_args(event_dir=events, save=ref), iteration=7)
        _write_tracker(ref, "7")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ref, requested_load=tmp_path / "run"))

        assert (events / "main.jsonl").read_text() == "fresh\n"

    def test_restore_skips_release_tracker(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A tracker containing the non-numeric text "release" is not a resumable
        # iteration; the live file is left unchanged.
        """A non-numeric tracker (e.g. 'release') is not a resumable iteration."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "release")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")

        event_logger_checkpoint.restore(_args(event_dir=events, load=ckpt))

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_restore_of_a_named_trainer_ignores_a_tracker_left_at_the_root(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A tracker (iteration 3) and an empty snapshot folder exist at the root
        # of the checkpoint directory, but the run has a named trainer config ("policy"), whose
        # tracker would live in its own namespace. Restore must ignore the root tracker and keep
        # the live file.
        """The cursor such a run rewinds to is its own, so a root tracker is some other layout's."""
        ckpt = tmp_path / "ckpt"
        _write_tracker(ckpt, "3")
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("keep\n")
        (ckpt / "iter_0000003" / "debug_events").mkdir(parents=True)

        event_logger_checkpoint.restore(
            _args(event_dir=events, load=ckpt, megatron_config=encode_megatron_config("policy"))
        )

        assert (events / "main.jsonl").read_text() == "keep\n"

    def test_snapshot_skips_when_events_disabled_or_no_save(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two calls: one with no event directory but a save directory, one with
        # an event directory but no save directory. Neither creates anything; the would-be
        # checkpoint directory does not exist afterwards.
        """Disabled events or no save dir means no snapshot."""
        events = tmp_path / "events"
        events.mkdir()

        event_logger_checkpoint.snapshot(_args(event_dir=None, save=tmp_path / "ckpt"), iteration=1)
        event_logger_checkpoint.snapshot(_args(event_dir=events), iteration=1)

        assert not (tmp_path / "ckpt").exists()


class TestDiscardEventLog:
    # [CLASS DOC] Tests for discard, which moves the live event directory's content to a hidden
    # trash directory (name starting with ".trash_") beside it and recreates an empty live
    # directory, used when one run takes over from another. It refuses symbolic links.
    def test_moves_the_log_aside_and_leaves_an_empty_directory(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] After discarding a directory with one file, exactly one trash folder
        # exists next to it holding the file with its content, and the live directory exists and
        # is empty.
        """The live loggers reopen this path on every write, so a directory that vanished crashes them."""
        events = tmp_path / "events"
        events.mkdir()
        (events / "main.jsonl").write_text("from the run being taken over\n")

        event_logger_checkpoint.discard(_args(event_dir=events))

        [trash] = list(tmp_path.glob(".trash_*"))
        assert (trash / "main.jsonl").read_text() == "from the run being taken over\n"
        assert events.is_dir() and list(events.iterdir()) == []

    def test_leaves_an_empty_directory_alone(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Discarding an already empty directory creates no trash folder and keeps
        # the directory.
        """A take-over of a run that logged nothing has nothing to throw away, and leaves no empty trash."""
        events = tmp_path / "events"
        events.mkdir()

        event_logger_checkpoint.discard(_args(event_dir=events))

        assert list(tmp_path.glob(".trash_*")) == []
        assert events.is_dir()

    def test_leaves_a_missing_directory_alone(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Discarding a directory that does not exist creates neither trash nor
        # the directory.
        """The run may be taken over before it opened its log at all."""
        events = tmp_path / "events"

        event_logger_checkpoint.discard(_args(event_dir=events))

        assert list(tmp_path.glob(".trash_*")) == []
        assert not events.exists()

    def test_refuses_a_live_symlink(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] When the live directory is a symbolic link to another directory,
        # discard raises an assertion error matching "is a symlink", and neither the link nor
        # the target's file is changed.
        """Moving the link aside would leave the loggers writing into a directory nobody reads."""
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / "main.jsonl").write_text("not this run's to move\n")
        events = tmp_path / "events"
        events.symlink_to(target)

        with pytest.raises(AssertionError, match="is a symlink"):
            event_logger_checkpoint.discard(_args(event_dir=events))

        assert events.is_symlink()
        assert (target / "main.jsonl").read_text() == "not this run's to move\n"

    def test_refuses_a_dangling_symlink(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A symbolic link whose target does not exist is also refused with the
        # same assertion error, and stays in place.
        """A dangling link is not a directory, so the emptiness check alone would pass it silently."""
        events = tmp_path / "events"
        events.symlink_to(tmp_path / "gone")

        with pytest.raises(AssertionError, match="is a symlink"):
            event_logger_checkpoint.discard(_args(event_dir=events))

        assert events.is_symlink()

    def test_leaves_a_run_that_logs_nowhere_alone(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With no event directory configured (none), discard does nothing and
        # the temporary directory stays empty.
        """Without --save-debug-event-data there is no log to move, and nothing may be created."""
        event_logger_checkpoint.discard(_args(event_dir=None))

        assert list(tmp_path.iterdir()) == []
