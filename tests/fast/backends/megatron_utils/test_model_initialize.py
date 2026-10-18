# [MODULE DOC]
# Tests for model and optimizer initialization and checkpoint-state loading in the Megatron model
# module. The real Megatron, SGLang debug-utils and a few miles modules are replaced in the module
# table by stubs for the duration of this file (autouse module-scoped fixture), so the model
# module can be imported on a CPU-only machine. Verifies how the learning-rate scheduler is
# stepped after loading, and where a load says the run starts (finetune, resume, LoRA adapters).
import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from miles.backends.megatron_utils.model import LoadCheckpointOutput


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    # [FUNCTION DOC] HELPER: Creates an empty module object with the given dotted name, optionally
    # marks it as a package (empty search path), sets each given attribute on it, registers it in
    # the global module table (replacing any real module of that name), and returns it. Side
    # effect on global interpreter state; the caller's fixture restores the table.
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    if attrs is not None:
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


class _DummyDDP:
    # [CLASS DOC] HELPER: Empty placeholder for the distributed data-parallel wrapper class.
    pass


class _DummyModel:
    # [CLASS DOC] HELPER: Empty placeholder for model classes (GPT model, last-layer linear).
    pass


class _DummyOptimizer:
    # [CLASS DOC] HELPER: Empty placeholder for optimizer classes (Adam, CPU Adam, Muon, base).
    pass


class _DummyChainedOptimizer:
    # [CLASS DOC] HELPER: Empty placeholder for the chained optimizer class.
    pass


class _DummyDistributedOptimizer:
    # [CLASS DOC] HELPER: Empty placeholder for the distributed optimizer class.
    pass


class _DummyScheduler:
    # [CLASS DOC] HELPER: Empty placeholder for the optimizer parameter scheduler class.
    pass


class _DummyOptimizerConfig:
    # [CLASS DOC] HELPER: Placeholder optimizer config that stores any keyword arguments as
    # attributes.
    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: Copies all keyword arguments into the instance attributes.
        self.__dict__.update(kwargs)


class _FakeModelChunk:
    # [CLASS DOC] HELPER: Fake model chunk with a class-level role attribute defaulting to none.
    role: str | None = None


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_environment():
    # [FUNCTION DOC] HELPER: Module-scoped autouse fixture. Snapshots the global module table,
    # installs stub modules for the Megatron core packages (parallel utilities, tensor-parallel
    # RNG helpers, distributed wrapper, enums, GPT model, optimizers and their sub-modules,
    # scheduler, packed-sequence params, pipeline schedule getter, transformer utilities, config,
    # microbatch calculator, training globals, training and checkpointing functions), the SGLang
    # debug dumper, and the miles LoRA bridge and model provider modules, using mocks and the
    # dummy classes above. Yields to the tests, then always clears and restores the original
    # module table so the stubs do not leak into other test files.
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron", is_package=True)
        core_module = _stub_module("megatron.core", is_package=True)
        core_module.mpu = types.SimpleNamespace()
        core_module.tensor_parallel = _stub_module(
            "megatron.core.tensor_parallel",
            {"model_parallel_cuda_manual_seed": MagicMock()},
            is_package=True,
        )
        _stub_module(
            "megatron.core.tensor_parallel.random",
            {"_get_all_rng_states": MagicMock(), "_set_all_rng_states": MagicMock()},
        )
        _stub_module(
            "megatron.core.distributed",
            {
                "DistributedDataParallel": _DummyDDP,
                "finalize_model_grads": MagicMock(),
            },
        )
        _stub_module(
            "megatron.core.enums",
            {"ModelType": types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")},
        )
        _stub_module("megatron.core.models", is_package=True)
        _stub_module("megatron.core.models.gpt", {"GPTModel": _DummyModel})
        _stub_module(
            "megatron.core.optimizer",
            {
                "OptimizerConfig": _DummyOptimizerConfig,
                "get_megatron_optimizer": MagicMock(),
                "Adam": _DummyOptimizer,
                "CPUAdam": _DummyOptimizer,
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.emerging_optimizers", {"TensorParallelMuon": _DummyOptimizer})
        _stub_module("megatron.core.optimizer.muon", {"get_megatron_muon_optimizer": MagicMock()})
        _stub_module("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": _DummyDistributedOptimizer})
        _stub_module(
            "megatron.core.optimizer.optimizer",
            {
                "ChainedOptimizer": _DummyChainedOptimizer,
                "MegatronOptimizer": _DummyOptimizer,
            },
        )
        _stub_module("megatron.core.optimizer_param_scheduler", {"OptimizerParamScheduler": _DummyScheduler})
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": MagicMock()})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock(), "unwrap_model": MagicMock()})
        _stub_module("megatron.core.config", {"set_experimental_flag": MagicMock()})
        _stub_module("megatron.core.num_microbatches_calculator", {"init_num_microbatches_calculator": MagicMock()})
        _stub_module("megatron.training", is_package=True)
        _stub_module(
            "megatron.training.global_vars",
            {
                "get_args": MagicMock(),
                "_build_tokenizer": MagicMock(),
                "set_args": MagicMock(),
            },
        )
        _stub_module("megatron.training.training", {"get_model": MagicMock()})
        _stub_module(
            "megatron.training.checkpointing",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
            },
        )
        _stub_module("sglang.srt.debug_utils", is_package=True)
        _stub_module(
            "sglang.srt.debug_utils.dumper",
            {
                "DumperConfig": MagicMock(),
                "_get_rank": MagicMock(return_value=0),
                "dumper": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.lora.bridge",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.model_provider",
            {
                "get_model_provider_func": MagicMock(),
                "LinearForLastLayer": _DummyModel,
            },
        )
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    # [FUNCTION DOC] HELPER: Enters, into the given exit stack, patches that replace three
    # side-effecting functions of the model module with mocks: memory clearing, the peak GPU
    # memory check after load, and the model hash check.
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    # [FUNCTION DOC] Arguments say the scheduler is restored from the checkpoint (global batch
    # size 8, no finetune). Setup is patched to return a fake model, optimizer and mock scheduler,
    # and the checkpoint loader returns (iteration 100, 0, False). Asserts the initializer returns
    # model, optimizer, scheduler and a load output with loaded rollout id 100 and start rollout
    # id 101, and that the scheduler was never stepped.
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=True, global_batch_size=8, finetune=False)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0, False)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (
        model,
        optimizer,
        opt_param_scheduler,
        LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101),
    )
    opt_param_scheduler.step.assert_not_called()


def test_initialize_steps_scheduler_when_checkpoint_did_not_restore_it():
    # [FUNCTION DOC] Same setup as the restored case but the arguments say the scheduler is not
    # restored from the checkpoint. Asserts the same return value and that the scheduler was
    # stepped exactly once with an increment of 800 (iteration 100 times global batch size 8).
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=8, finetune=False)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0, False)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (
        model,
        optimizer,
        opt_param_scheduler,
        LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101),
    )
    opt_param_scheduler.step.assert_called_once_with(increment=800)


def _load_model_state_with(
    *, tmp_path: Path, finetune: bool, iteration: int, lora_rank: int = 0
) -> "LoadCheckpointOutput":
    # [FUNCTION DOC] HELPER: Keyword-only. Creates a checkpoint directory under the temporary
    # path containing a tracker file with the given iteration, patches the checkpoint loader to
    # return (iteration, 0, False) plus the side-effect patches, and calls the model-state loader
    # with arguments for the given finetune flag and LoRA rank (default 0), core conversion mode,
    # no adapter path, a fake model chunk, no optimizer or scheduler, actor role and no
    # checkpointing context. Returns the load output; propagates the loader's assertion errors.
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir()
    (load_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    with ExitStack() as stack:
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(iteration, 0, False))
        )
        _patch_initialize_side_effects(stack)
        return load_model_state(
            Namespace(
                use_checkpoint_opt_param_scheduler=True,
                global_batch_size=8,
                finetune=finetune,
                lora_rank=lora_rank,
                megatron_to_hf_mode="core",
                lora_adapter_path=None,
                load=str(load_dir),
            ),
            model=[_FakeModelChunk()],
            optimizer=None,
            opt_param_scheduler=None,
            role="actor",
            checkpointing_context=None,
        )


class TestWhereALoadSaysTheRunStarts:
    # [CLASS DOC] Tests for the start rollout id derived by the model-state loader.
    def test_a_finetune_load_starts_the_run_at_rollout_zero(self, tmp_path: Path):
        # [FUNCTION DOC] Finetune with iteration 0: start rollout id is 0.
        """--finetune means there is no run to continue, so rollout 0 is still ahead rather than behind."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=0).start_rollout_id == 0

    def test_a_resumed_load_starts_the_run_after_the_checkpoint_it_read(self, tmp_path: Path):
        # [FUNCTION DOC] No finetune, iteration 100: start rollout id is 101.
        """The checkpoint's own rollout is done, so the run continues at the next one."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=100).start_rollout_id == 101

    def test_a_run_that_restored_the_iteration_zero_checkpoint_it_wrote_starts_at_one(self, tmp_path: Path):
        # [FUNCTION DOC] No finetune, iteration 0: start rollout id is 1 (a resume, not a restart).
        """A real resume from the very first checkpoint must not be read as a finetune that starts over."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=0).start_rollout_id == 1

    def test_a_finetune_load_that_found_a_checkpoint_is_refused(self, tmp_path: Path):
        # [FUNCTION DOC] Finetune with iteration 100: raises an assertion error saying the two
        # disagree about where this run stands.
        """--finetune promises iteration 0; anything else means the two disagree about where the run stands."""
        with pytest.raises(AssertionError, match="disagree about where this run stands"):
            _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=100)


class TestALoraAdapterThatCarriesItsOwnIteration:
    # [CLASS DOC] Tests for LoRA runs, whose adapter checkpoints carry their own iteration and
    # write no tracker file, so resumes arrive with finetune set.
    def test_a_lora_resume_under_finetune_continues_after_the_iteration_the_adapter_names(self, tmp_path: Path):
        # [FUNCTION DOC] Finetune, LoRA rank 8, loader-reported iteration 100: start rollout id is
        # 101 (the finetune/iteration disagreement check does not apply to LoRA).
        """LoRA saves write no tracker, so a lora resume always arrives here with --finetune set."""
        output = _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=100, lora_rank=8)

        assert output.start_rollout_id == 101

    def test_a_lora_run_that_really_starts_from_scratch_still_starts_at_rollout_one(self, tmp_path: Path):
        # [FUNCTION DOC] Finetune, LoRA rank 8, iteration 0: start rollout id is 1.
        """An adapter with no training state answers iteration 0, and the run continues from the next rollout."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=0, lora_rank=8).start_rollout_id == 1
