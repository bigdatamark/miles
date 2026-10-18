# [MODULE DOC] CPU unit tests of the Megatron train actor's reload-state operation (the "hot
# restart" that rolls a live trainer back to the checkpoint the run was told to load from, or to
# its cold-start state when nothing was saved yet). The actor class is created without its
# constructor and given hand-set fields; the checkpoint loader and memory helpers in the model
# module are replaced with recorders that snapshot the run arguments seen during the load. Covered:
# the scheduler is reset to the start; the load directory is the requested one (not a cold-start
# fallback), the previous rollout id is forgotten, cold-start flags are cleared for the load and
# restored afterwards, the resume rollout is iteration plus one, critics read their own directory;
# a cold-start with no saved checkpoint resets the trainer using the original arguments and restores
# the random state and optimizer; a long list of unsupported configurations is refused with
# assertion errors; pending async saves are finalized first. The actor module is imported through a
# module-scoped fixture that stubs the memory-saver dependency and restores the module table.
import importlib
import logging
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple
from unittest.mock import Mock

import pytest

from miles.utils.init_once import InitOnce

_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


class _CheckpointLoadResult(NamedTuple):
    # [CLASS DOC] HELPER: Immutable stand-in for the checkpoint loader's return value with three
    # fields: the loaded iteration number, the cumulative floating-point operation count (zero here),
    # and whether the native optimizer state was restored.
    iteration: int
    num_floating_point_operations_so_far: int
    native_optimizer_restored: bool


class _FakeRandomState:
    # [CLASS DOC] HELPER: Stand-in for a saved random-number-generator state that counts how many
    # times it was restored.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts the restore counter at zero.
        self.restores = 0

    def restore(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the restore counter.
        self.restores += 1


@pytest.fixture(scope="module")
def actor_module():
    # [FUNCTION DOC] HELPER: Module-scoped generator fixture. Setup: remembers the current module
    # table entries for the actor module and the memory-saver module and the package's actor attribute,
    # installs a fake memory-saver module with a mock saver, removes the cached actor module and package
    # attribute, then imports the actor module; if the import fails with an import error the whole
    # fixture skips the dependent tests with the error text. Teardown (always): drops the imported
    # module and restores or removes the earlier table entries and package attribute.
    """Import the megatron actor with the one native dependency the fast environment lacks stubbed out."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        try:
            module = importlib.import_module(_ACTOR_MODULE_NAME)
        except ImportError as e:
            pytest.skip(f"the megatron actor cannot be imported here: {e!r}")
        yield module
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


def _args(tmp_path: Path, **overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a run-arguments namespace for a normal (non-cold-start) run with
    # every field the reload path reads: load and requested-load both the "pretrain" directory under the
    # temp path, save directory "run", no step overrides, load flags off, finetune off, core
    # conversion mode, no LoRA, no colocation, no async save, no offload or profiling, plain Adam
    # optimizer settings and global batch size 1. Keyword overrides replace defaults (unknown names are
    # added). Returns the namespace.
    defaults = dict(
        load=str(tmp_path / "pretrain"),
        requested_load=str(tmp_path / "pretrain"),
        save=str(tmp_path / "run"),
        ckpt_step=None,
        ref_ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        megatron_to_hf_mode="core",
        ref_update_interval=None,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
        rematerialize_param_from_master_weight=False,
        non_persistent_ckpt_type=None,
        fp16=False,
        use_precision_aware_optimizer=False,
        optimizer="adam",
        optimizer_cpu_offload=False,
        offload_optimizer_states=False,
        chunked_optimizer_state_offload=False,
        debug_rollout_only=False,
        async_save=False,
        offload_train=False,
        stream_optimizer_state_to_disk=False,
        keep_old_actor=False,
        use_pytorch_profiler=False,
        record_memory_history=False,
        use_checkpoint_opt_param_scheduler=True,
        global_batch_size=1,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_checkpoint(directory: Path, *, iteration: int) -> str:
    # [FUNCTION DOC] HELPER: Creates the directory (with parents) and writes the latest-iteration
    # marker file containing the iteration and a newline, making it look like a saved checkpoint
    # directory. Returns the directory path as a string.
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    return str(directory)


def _inited_guard() -> InitOnce:
    # [FUNCTION DOC] HELPER: Returns an initialize-once guard for the actor class name that has
    # already been entered once (so the actor looks initialized).
    guard = InitOnce("MegatronTrainRayActor")
    with guard.guarding():
        pass
    return guard


def _actor(actor_module, *, role: str, args: Namespace):
    # [FUNCTION DOC] HELPER: Creates a Megatron train actor without running its constructor and sets
    # the fields the reload path touches: role, args, an already-entered init guard, awake, no
    # reference or teacher model, no model, optimizer or scheduler, no last rollout id, and a fake
    # post-init random state. Returns the actor.
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.role = role
    actor.args = args
    actor._init_once = _inited_guard()
    actor._asleep = False
    actor.with_ref = False
    actor.with_opd_teacher = False
    actor.model = None
    actor.optimizer = None
    actor.opt_param_scheduler = None
    actor._last_rollout_id = None
    actor._post_init_random_state = _FakeRandomState()
    return actor


def _watch_load(actor_module, monkeypatch, *, args: Namespace, iteration: int) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Replaces, via the monkeypatch fixture, the model module's checkpoint
    # loader with a recorder that stores a copy of the run arguments' fields as they are during the
    # load and returns a result with the given iteration; and replaces the memory clearing, peak-memory
    # check and model-hash check helpers (and the actor module's memory clearing) with no-ops. Returns
    # the dictionary into which the recorder stores the argument snapshot under "args_during_load".
    model_module = importlib.import_module("miles.backends.megatron_utils.model")
    seen: dict[str, Any] = {}

    def fake_load_checkpoint(*_args: Any, **_kwargs: Any) -> _CheckpointLoadResult:
        # [FUNCTION DOC] HELPER: Fake loader ignoring its inputs; snapshots the run arguments into
        # the shared dictionary and returns a result carrying the chosen iteration, zero operations
        # and native optimizer not restored.
        seen["args_during_load"] = vars(args).copy()
        return _CheckpointLoadResult(
            iteration=iteration, num_floating_point_operations_so_far=0, native_optimizer_restored=False
        )

    monkeypatch.setattr(model_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(model_module, "clear_memory", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda *a, **k: None)
    monkeypatch.setattr(actor_module, "clear_memory", lambda *a, **k: None)
    return seen


class _Scheduler:
    # [CLASS DOC] HELPER: Minimal stand-in for an optimizer parameter scheduler tracking only a step
    # counter.
    def __init__(self, *, num_steps: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the initial step count.
        self.num_steps = num_steps

    def step(self, *, increment: int) -> None:
        # [FUNCTION DOC] HELPER: Adds the (keyword-only) increment to the step count.
        self.num_steps += increment


class TestLoadStateScheduler:
    # [CLASS DOC] Tests of what the reload does to the live parameter scheduler.
    @pytest.mark.parametrize("saved_checkpoint", [True, False])
    def test_a_reload_leaves_the_live_scheduler_at_the_start(
        self,
        actor_module: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        saved_checkpoint: bool,
    ) -> None:
        # [FUNCTION DOC] Parametrized over a run that has a saved checkpoint (iteration 50) and a cold
        # started run with none (iteration 0, with the optimizer reset stubbed out). Gives the actor a
        # scheduler that has already taken 37 steps, reloads state, and asserts the scheduler's step
        # count is zero afterwards.
        """A reload must leave no scheduler progress from the trainer's previous run."""
        if saved_checkpoint:
            _write_checkpoint(tmp_path / "pretrain", iteration=50)
            args = _args(tmp_path)
            iteration = 50
        else:
            args = _cold_started_args(tmp_path)
            iteration = 0
            _stub_reset(actor_module, monkeypatch)
        _watch_load(actor_module, monkeypatch, args=args, iteration=iteration)
        actor = _actor(actor_module, role="actor", args=args)
        scheduler = _Scheduler(num_steps=37)
        actor.opt_param_scheduler = scheduler

        actor.load_state()

        assert scheduler.num_steps == 0


class TestTheCheckpointAReloadRollsBackTo:
    # [CLASS DOC] Tests of which checkpoint the reload reads and how it treats the run arguments
    # around the load.
    def test_a_reload_reads_the_directory_the_run_was_told_to_load_from(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a saved checkpoint at iteration 50 in the requested directory, asserts
        # reload returns 51 (iteration plus one, the rollout to resume at) and that the load path seen
        # by the loader equals that directory.
        """The data source reads its own state out of that same directory, so the two have to be the same one."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        assert _actor(actor_module, role="actor", args=args).load_state() == 51
        assert seen["args_during_load"]["load"] == load

    def test_a_reload_reads_what_the_run_asked_for_rather_than_what_a_parse_fell_back_to(
        self, actor_module, tmp_path, monkeypatch
    ):
        # [FUNCTION DOC] The run arguments' load field was rewritten to a reference-weights path (as
        # a cold-started parse does) while the requested load is the saved directory; asserts the load
        # path seen by the loader is the requested saved directory, not the rewritten one.
        """A cold-started parse rewrites `--load` to the reference weights, which a reload must not read."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, load=str(tmp_path / "reference"), finetune=True)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        _actor(actor_module, role="actor", args=args).load_state()

        assert seen["args_during_load"]["load"] == load

    def test_a_reload_forgets_the_rollout_the_previous_script_was_last_on(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] Sets the actor's last rollout id to 49, reloads, and asserts it is none again.
        """A freshly started trainer has not trained yet, and a reloaded one has to look the same to the next one."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path)
        _watch_load(actor_module, monkeypatch, args=args, iteration=50)
        actor = _actor(actor_module, role="actor", args=args)
        actor._last_rollout_id = 49

        actor.load_state()

        assert actor._last_rollout_id is None

    def test_a_reload_clears_the_flags_a_cold_started_parse_left_behind(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] Arguments carry cold-start leftovers (finetune on, optimizer and RNG loading
        # disabled, a checkpoint step override of 3). Asserts that during the load the loader sees
        # finetune off, both loading-disabled flags off, and the step override cleared (none).
        """A parse that found no checkpoint set these, and a reload onto a real checkpoint has to load all of it."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=3)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        _actor(actor_module, role="actor", args=args).load_state()

        during = seen["args_during_load"]
        assert (during["finetune"], during["no_load_optim"], during["no_load_rng"], during["ckpt_step"]) == (
            False,
            False,
            False,
            None,
        )

    def test_a_cold_started_parse_does_not_make_a_real_resume_start_over(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] With cold-start flags set in the arguments but a real checkpoint at iteration
        # 50 saved, asserts reload still returns 51, i.e. the resume rollout is computed under the
        # overridden (cleared) arguments, not the cold-start ones.
        """The rollout to resume at is worked out under the overridden arguments, not the restored ones."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True)
        _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        assert _actor(actor_module, role="actor", args=args).load_state() == 51

    def test_a_reload_leaves_the_arguments_as_it_found_them(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] After reload with cold-start-style arguments, asserts the arguments object is
        # restored: load path unchanged, and finetune, no-load-optim, no-load-rng all true and step
        # override 3.
        """The override says where this one load reads from; the run's own arguments have to survive it."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=3)
        _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        _actor(actor_module, role="actor", args=args).load_state()

        assert args.load == str(tmp_path / "pretrain")
        assert (args.finetune, args.no_load_optim, args.no_load_rng, args.ckpt_step) == (True, True, True, 3)

    def test_a_critic_reload_reads_the_critic_directory(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] For role critic with the requested load pointing at a separate critic
        # checkpoint directory (iteration 60), asserts reload returns 61 and the loader saw that
        # directory.
        """A critic's own arguments carry its checkpoint dirs, so reading them reads the critic's."""
        critic_load = _write_checkpoint(tmp_path / "pretrain_critic", iteration=60)
        args = _args(tmp_path, requested_load=critic_load)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=60)

        assert _actor(actor_module, role="critic", args=args).load_state() == 61
        assert seen["args_during_load"]["load"] == critic_load

    def test_a_reload_that_would_cold_start_is_refused(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a requested load that holds no checkpoint (the not-yet-written save
        # directory) and a loader reporting iteration 0, asserts reload raises an assertion error
        # (the cold-start reset path is not allowed for this argument set because the cold-start
        # flags are absent).
        """The trainer is alive at the rollout the run reached, so a run that cold started is refused here."""
        args = _args(tmp_path, requested_load=str(tmp_path / "run"))
        _watch_load(actor_module, monkeypatch, args=args, iteration=0)

        with pytest.raises(AssertionError):
            _actor(actor_module, role="actor", args=args).load_state()


def _reference_weights(tmp_path: Path) -> str:
    # [FUNCTION DOC] HELPER: Creates a "reference" directory under the temp path containing an empty
    # JSON config file (looking like a hugging-face model directory) and returns its path string.
    directory = tmp_path / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    return str(directory)


def _cold_started_args(tmp_path: Path, **overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Like the normal arguments builder but modelling a cold-started run: the
    # load field points at reference weights (a hugging-face style directory), the requested load is
    # the not-yet-written "run" directory, and finetune plus both load-disabled flags are on. Keyword
    # overrides win. Returns the namespace.
    defaults = dict(
        load=_reference_weights(tmp_path),
        requested_load=str(tmp_path / "run"),
        finetune=True,
        no_load_optim=True,
        no_load_rng=True,
    )
    defaults.update(overrides)
    return _args(tmp_path, **defaults)


def _stub_reset(actor_module, monkeypatch) -> None:
    # [FUNCTION DOC] HELPER: Replaces the actor module's optimizer-state reset function with a no-op
    # for the duration of the test.
    monkeypatch.setattr(actor_module, "reset_optimizer_state", lambda *a, **k: None)


class TestAReloadThatFindsNothingItSaved:
    # [CLASS DOC] Tests of the cold-start path: the run has saved nothing yet, so reload returns the
    # trainer to its starting state from the original (reference) weights.
    def test_it_puts_the_trainer_back_where_the_run_began(self, actor_module, tmp_path, monkeypatch, caplog):
        # [FUNCTION DOC] With cold-started arguments and a loader reporting iteration 0, reloads at
        # info log level and asserts it returns 0, the loader read the reference-weights directory, and
        # the log mentions that no checkpoint was found.
        """A run that has not saved yet still has a state to go back to: the one it was started from."""
        args = _cold_started_args(tmp_path)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=0)
        actor = _actor(actor_module, role="actor", args=args)
        _stub_reset(actor_module, monkeypatch)

        with caplog.at_level(logging.INFO):
            assert actor.load_state() == 0

        assert seen["args_during_load"]["load"] == _reference_weights(tmp_path)
        assert "found no checkpoint" in caplog.text

    def test_it_loads_under_the_arguments_the_run_was_started_with(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] With cold-started arguments (checkpoint step overrides 3), reloads and asserts
        # the loader saw the cold-start flags unchanged: finetune on, both load-disabled flags on, and
        # step override 3.
        """Anything else would answer a different question than the one the trainer's own init answered."""
        args = _cold_started_args(tmp_path, ckpt_step=3, ref_ckpt_step=3)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=0)
        actor = _actor(actor_module, role="actor", args=args)
        _stub_reset(actor_module, monkeypatch)

        actor.load_state()

        during = seen["args_during_load"]
        assert (during["finetune"], during["no_load_optim"], during["no_load_rng"], during["ckpt_step"]) == (
            True,
            True,
            True,
            3,
        )

    def test_it_restores_the_random_stream_and_resets_the_optimizer(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] Gives the actor a placeholder live optimizer and replaces the optimizer reset
        # with a recorder (keyword-only stream-to-disk and chunked-offload parameters) that appends the
        # optimizer it was given to a list. After reload asserts the fake post-init random state was
        # restored exactly once and the reset was called exactly once with the live optimizer.
        """A checkpoint load overwrites neither, so a live trainer would keep both from the rollouts it discards."""
        args = _cold_started_args(tmp_path)
        _watch_load(actor_module, monkeypatch, args=args, iteration=0)
        actor = _actor(actor_module, role="actor", args=args)
        actor.optimizer = "the live optimizer"
        reset: list[str] = []
        monkeypatch.setattr(
            actor_module,
            "reset_optimizer_state",
            lambda optimizer, *, stream_optimizer_state_to_disk, chunked_optimizer_state_offload: reset.append(
                optimizer
            ),
        )

        actor.load_state()

        assert actor._post_init_random_state.restores == 1 and reset == ["the live optimizer"]

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(finetune=False),
            dict(no_load_optim=False),
            dict(no_load_rng=False),
            dict(ckpt_step=7),
        ],
    )
    def test_a_run_that_did_not_cold_start_is_refused(self, actor_module, tmp_path, monkeypatch, overrides):
        # [FUNCTION DOC] Parametrized over turning off finetune, turning off either load-disabled flag
        # or setting a step override, each on top of a cold-started argument set with no saved
        # checkpoint. Asserts reload raises an assertion error.
        """Only a cold-started run is supported here; resetting state a load would also restore is undesigned."""
        args = _cold_started_args(tmp_path, **overrides)
        _watch_load(actor_module, monkeypatch, args=args, iteration=0)
        _stub_reset(actor_module, monkeypatch)

        with pytest.raises(AssertionError):
            _actor(actor_module, role="actor", args=args).load_state()

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(fp16=True),
            dict(use_precision_aware_optimizer=True),
            dict(optimizer_cpu_offload=True),
            dict(offload_optimizer_states=True),
        ],
    )
    def test_a_trainer_whose_state_cannot_be_reset_is_refused(self, actor_module, tmp_path, overrides):
        # [FUNCTION DOC] Parametrized over fp16, precision-aware optimizer, optimizer CPU offload and
        # optimizer-state offload, each applied to cold-started arguments. Asserts reload raises an
        # assertion error (nothing is stubbed, so it must refuse before reading or loading).
        """These keep state a reset would have to rebuild, and a half-reset trainer is worse than a refused one."""
        actor = _actor(actor_module, role="actor", args=_cold_started_args(tmp_path, **overrides))

        with pytest.raises(AssertionError):
            actor.load_state()

    def test_a_reload_that_did_save_is_not_pushed_back_to_the_beginning(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a saved checkpoint at iteration 50 and fp16 turned on (which would
        # forbid the cold-start reset path), asserts reload succeeds and returns 51, i.e. the reset
        # restrictions apply only on the cold-start path.
        """The reset path is for a run with nothing of its own, and running it over a real resume would undo it."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, fp16=True)
        _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        assert _actor(actor_module, role="actor", args=args).load_state() == 51

    def test_a_bridge_run_without_a_checkpoint_is_refused(self, actor_module, tmp_path, monkeypatch):
        # [FUNCTION DOC] With cold-started arguments in bridge conversion mode, asserts reload raises
        # an assertion error whose message says bridge mode is unsupported.
        """Bridge initialization cannot recreate the cold-start state before the run's first save."""
        args = _cold_started_args(tmp_path, megatron_to_hf_mode="bridge")
        _watch_load(actor_module, monkeypatch, args=args, iteration=0)

        with pytest.raises(AssertionError, match="bridge mode unsupported"):
            _actor(actor_module, role="actor", args=args).load_state()

    def test_an_async_save_still_in_flight_is_finalized_before_the_choice_is_made(
        self, actor_module, tmp_path, monkeypatch
    ):
        # [FUNCTION DOC] With async saving on and the requested load pointing at a directory not yet
        # written, replaces the actor class's finalize-pending-save method with one that writes the
        # checkpoint (iteration 50) into that directory. Asserts reload returns 51 and the loader read
        # that directory, proving finalization happens before the checkpoint choice.
        """The checkpoint a pending save is about to write is the one the reload has to come back to."""
        args = _args(tmp_path, async_save=True, requested_load=str(tmp_path / "run"))
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=50)
        actor = _actor(actor_module, role="actor", args=args)
        monkeypatch.setattr(
            type(actor),
            "_finalize_pending_async_save",
            lambda self: _write_checkpoint(tmp_path / "run", iteration=50),
        )

        assert actor.load_state() == 51
        assert seen["args_during_load"]["load"] == str(tmp_path / "run")


class TestWhatAReloadRefuses:
    # [CLASS DOC] Tests of the configurations for which reload raises an assertion error.
    @pytest.mark.parametrize(
        "overrides",
        [
            dict(debug_rollout_only=True),
            dict(lora_rank=8),
            dict(multi_lora=True),
            dict(colocate=True),
            dict(rematerialize_param_from_master_weight=True),
            dict(non_persistent_ckpt_type="local"),
            dict(offload_train=True),
            dict(use_pytorch_profiler=True),
            dict(record_memory_history=True),
            dict(keep_old_actor=True),
        ],
    )
    def test_a_trainer_it_cannot_restore_is_refused_before_anything_is_read(self, actor_module, tmp_path, overrides):
        # [FUNCTION DOC] Parametrized over ten unsupported settings (rollout-only debug, LoRA rank,
        # multi-LoRA, colocation, rematerializing params from master weights, non-persistent
        # checkpoints, train offload, profiler, memory history, keep-old-actor). With no checkpoint
        # written and no stubs, asserts reload raises an assertion error.
        """Each of these means a checkpoint load cannot put the trainer back where the run really is."""
        actor = _actor(actor_module, role="actor", args=_args(tmp_path, **overrides))

        with pytest.raises(AssertionError):
            actor.load_state()

    def test_a_run_holding_a_second_copy_of_the_actor_says_why_it_is_refused(self, actor_module, tmp_path):
        # [FUNCTION DOC] With keep-old-actor on (and a saved checkpoint present), asserts the assertion
        # error message mentions a second copy of the actor.
        """`--keep-old-actor` is the one refusal whose reason is not obvious from reading the flag's name."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        actor = _actor(actor_module, role="actor", args=_args(tmp_path, keep_old_actor=True))

        with pytest.raises(AssertionError, match="second copy of the actor"):
            actor.load_state()

    def test_a_run_that_was_never_given_a_load_directory_is_refused(self, actor_module, tmp_path):
        # [FUNCTION DOC] With the requested load unset (none), asserts the assertion error message
        # says a hot restart needs the load option.
        """There is no checkpoint for this reload to restore, and starting the trainer over would replay the run."""
        actor = _actor(actor_module, role="actor", args=_args(tmp_path, requested_load=None))

        with pytest.raises(AssertionError, match="a hot restart needs --load"):
            actor.load_state()

    def test_a_periodically_updated_reference_is_refused(self, actor_module, tmp_path):
        # [FUNCTION DOC] With a reference update interval of 5 and the actor holding a reference
        # model, asserts the assertion error message says no checkpoint holds it.
        """Reloading only the actor would pair it with reference weights from a later rollout."""
        actor = _actor(actor_module, role="actor", args=_args(tmp_path, ref_update_interval=5))
        actor.with_ref = True

        with pytest.raises(AssertionError, match="no checkpoint holds it"):
            actor.load_state()
