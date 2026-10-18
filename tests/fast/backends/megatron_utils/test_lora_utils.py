# [MODULE DOC]
# Tests for the Megatron LoRA utilities: detecting whether LoRA is enabled from arguments,
# recognizing LoRA weight names (HuggingFace style) and adapter parameter names (Megatron style),
# the adapter-name constant, strict loading of per-rank adapter shard files, and saving/loading of
# the training state (optimizer, scheduler, iteration) next to the adapter. Parallel state is
# faked as a single rank 0 and files go to pytest temporary directories.
"""LoRA detection, adapter parameters, and training checkpoint state."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.lora.utils as lora_utils
from miles.backends.megatron_utils.lora.utils import (
    _is_adapter_param_name,
    is_lora_enabled,
    load_lora_adapter,
    save_lora_checkpoint,
)
from miles.utils.lora.utils import LORA_ADAPTER_NAME, is_lora_weight_name

# ---------------------------------------------------------------------------
# is_lora_enabled
# ---------------------------------------------------------------------------


class TestIsLoraEnabled:
    # [CLASS DOC] Tests for the LoRA-enabled predicate: true if a positive rank or an adapter path
    # is set; missing attributes count as disabled.
    def test_enabled_by_rank(self):
        # [FUNCTION DOC] Rank 32, no adapter path: enabled.
        args = Namespace(lora_rank=32, lora_adapter_path=None)
        assert is_lora_enabled(args) is True

    def test_enabled_by_adapter_path(self):
        # [FUNCTION DOC] Rank 0 with an adapter path: enabled.
        args = Namespace(lora_rank=0, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_enabled_by_both(self):
        # [FUNCTION DOC] Rank 16 and an adapter path: enabled.
        args = Namespace(lora_rank=16, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_disabled(self):
        # [FUNCTION DOC] Rank 0 and no adapter path: disabled.
        args = Namespace(lora_rank=0, lora_adapter_path=None)
        assert is_lora_enabled(args) is False

    def test_disabled_missing_attrs(self):
        # [FUNCTION DOC] An arguments namespace with neither attribute: disabled, no error.
        args = Namespace()
        assert is_lora_enabled(args) is False


# ---------------------------------------------------------------------------
# is_lora_weight_name / _is_adapter_param_name
# ---------------------------------------------------------------------------


class TestIsLoraWeightName:
    # [CLASS DOC] Tests for the shared predicate recognizing HuggingFace-style LoRA weight names.
    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_A.default.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_B.default.weight",
        ],
    )
    def test_positive(self, name):
        # [FUNCTION DOC] Parametrized names containing a LoRA A or B weight (with or without the
        # default adapter segment and base-model prefix) are recognized.
        assert is_lora_weight_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    def test_negative(self, name):
        # [FUNCTION DOC] Parametrized ordinary base-model weight names (attention, embedding,
        # output head, MLP) are not recognized.
        assert is_lora_weight_name(name) is False


class TestIsAdapterParamName:
    # [CLASS DOC] Tests for the private predicate recognizing Megatron-style adapter parameters.
    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.lora_A.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
        ],
    )
    def test_positive(self, name):
        # [FUNCTION DOC] Parametrized names with a LoRA A weight or an adapter input/output linear
        # weight under a module path are recognized as adapter parameters.
        assert _is_adapter_param_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.weight",
            "module.decoder.layers.0.mlp.linear_fc1.weight",
            "module.embedding.word_embeddings.weight",
        ],
    )
    def test_negative(self, name):
        # [FUNCTION DOC] Parametrized base Megatron weight names (qkv, fc1, embedding) are not
        # recognized as adapter parameters.
        assert _is_adapter_param_name(name) is False


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    # [FUNCTION DOC] Pins the shared adapter name constant to the literal string "miles_lora".
    assert LORA_ADAPTER_NAME == "miles_lora"


class _AdapterModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny module with a base 2-by-2 linear layer and two zero-initialized
    # adapter parameters named lora_A (1 by 2) and lora_B (2 by 1).
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the linear layer and the two adapter parameters.
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Parameter(torch.zeros(1, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))


def _single_rank(monkeypatch):
    # [FUNCTION DOC] HELPER: Patches the parallel-state getter in the LoRA utils module so tensor
    # and pipeline parallel ranks are both 0 (undone by the monkeypatch fixture at test end).
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
    )


def test_load_lora_adapter_rejects_a_shard_that_does_not_match_the_adapter(tmp_path, monkeypatch):
    # [FUNCTION DOC] Saves a rank-0 shard holding lora_A plus a stale extra key but not lora_B.
    # Asserts loading into the adapter model raises a runtime error whose message lists the
    # missing key lora_B and the unexpected key stale_lora_C.
    """A silently partial load resumes training on a half-initialized adapter."""
    _single_rank(monkeypatch)
    torch.save({"lora_A": torch.ones(1, 2), "stale_lora_C": torch.ones(1)}, tmp_path / "adapter_megatron_rank0.pt")

    with pytest.raises(RuntimeError, match=r"missing=\['lora_B'\], unexpected=\['stale_lora_C'\]"):
        lora_utils.load_lora_adapter([_AdapterModel()], str(tmp_path))


def test_load_lora_adapter_rejects_shards_saved_under_another_layout(tmp_path, monkeypatch):
    # [FUNCTION DOC] Only a rank-1 shard exists while the process is rank 0. Asserts loading raises
    # a file-not-found error saying there is none for global rank 0.
    """Falling through to fresh adapter weights would hide a resharding mistake."""
    _single_rank(monkeypatch)
    torch.save({"lora_A": torch.ones(1, 2)}, tmp_path / "adapter_megatron_rank1.pt")

    with pytest.raises(FileNotFoundError, match="none for global rank 0"):
        lora_utils.load_lora_adapter([_AdapterModel()], str(tmp_path))


class TestSaveLoraCheckpointTrainingState:
    # [CLASS DOC] Tests for the training-state file written by the LoRA checkpoint save.
    def _save(self, tmp_path, *, no_save_optim, scheduler=None):
        # [FUNCTION DOC] HELPER: Saves a LoRA checkpoint into a checkpoint subdirectory of the
        # temporary path using a publisher whose adapter-write is a no-op, one fake model whose
        # only parameter is an adapter weight, bridge mode, the given no-save-optimizer flag, an
        # optimizer whose state is a step of 7, the optional scheduler, and iteration 3. Returns
        # the sorted file names in the checkpoint directory.
        publisher = SimpleNamespace(write_adapter=lambda *_: None)

        adapter = torch.nn.Parameter(torch.ones(2))
        model = [SimpleNamespace(named_parameters=lambda: [("layers.0.self_attention.lora_A.weight", adapter)])]
        args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=no_save_optim)
        optimizer = SimpleNamespace(state_dict=lambda: {"step": 7})
        save_lora_checkpoint(
            model,
            args,
            str(tmp_path / "checkpoint"),
            publisher=publisher,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            iteration=3,
        )
        return sorted(path.name for path in (tmp_path / "checkpoint").iterdir())

    @staticmethod
    def _state(tmp_path):
        # [FUNCTION DOC] HELPER: Loads and returns the rank-0 training-state file from the
        # checkpoint subdirectory, with unrestricted unpickling.
        return torch.load(tmp_path / "checkpoint" / "training_state_rank0.pt", weights_only=False)

    def test_training_state_is_written_by_default(self, tmp_path):
        # [FUNCTION DOC] Saving with optimizer saving on and a scheduler (state: lr 0.5). Asserts
        # the directory has exactly the adapter shard and training-state file for rank 0, and the
        # state holds the optimizer state, the scheduler state and iteration 3.
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, no_save_optim=False, scheduler=scheduler)

        assert files == ["adapter_megatron_rank0.pt", "training_state_rank0.pt"]
        state = self._state(tmp_path)
        assert state["optimizer"] == {"step": 7}
        assert state["opt_param_scheduler"] == {"lr": 0.5}
        assert state["iteration"] == 3

    def test_no_save_optim_drops_the_optimizer_and_keeps_the_resume_metadata(self, tmp_path):
        # [FUNCTION DOC] Saving with the no-save-optimizer flag set. Asserts the same two files are
        # written, the stored optimizer entry is none, and the scheduler state and iteration are
        # still stored.
        """--no-save-optim is about optimizer state; losing the step and the LR schedule with it
        would silently restart a resumed run from iteration 0."""
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, no_save_optim=True, scheduler=scheduler)

        assert files == ["adapter_megatron_rank0.pt", "training_state_rank0.pt"]
        state = self._state(tmp_path)
        assert state["optimizer"] is None
        assert state["opt_param_scheduler"] == {"lr": 0.5}
        assert state["iteration"] == 3


class TestLoadTrainingState:
    # [CLASS DOC] Tests for the private training-state loader, which returns the stored iteration
    # and whether the optimizer state was restored, and feeds states to optimizer and scheduler.
    @staticmethod
    def _recorder():
        # [FUNCTION DOC] HELPER: Returns a pair of an empty list and a stand-in object whose
        # load-state-dict method appends each received state to that list.
        loaded = []
        return loaded, SimpleNamespace(load_state_dict=loaded.append)

    def _write(self, tmp_path, optimizer_state):
        # [FUNCTION DOC] HELPER: Writes the rank-0 training-state file with iteration 3, the given
        # optimizer state and a scheduler state of lr 0.5.
        torch.save(
            {"iteration": 3, "optimizer": optimizer_state, "opt_param_scheduler": {"lr": 0.5}},
            tmp_path / "training_state_rank0.pt",
        )

    def test_an_optimizer_free_checkpoint_still_restores_the_step_and_the_schedule(self, tmp_path):
        # [FUNCTION DOC] Stored optimizer state is none. Asserts the loader returns iteration 3
        # with optimizer-restored false, the optimizer received nothing, and the scheduler
        # received its stored state.
        self._write(tmp_path, None)
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == (3, False)
        assert optimizer_loads == []
        assert scheduler_loads == [{"lr": 0.5}]

    def test_a_full_checkpoint_restores_the_optimizer(self, tmp_path):
        # [FUNCTION DOC] Stored optimizer state is a step of 7. Asserts the loader returns
        # iteration 3 with optimizer-restored true and both optimizer and scheduler received
        # their stored states.
        self._write(tmp_path, {"step": 7})
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == (3, True)
        assert optimizer_loads == [{"step": 7}]
        assert scheduler_loads == [{"lr": 0.5}]


class TestLoadTrainingStateOptimizerGate:
    # [CLASS DOC] Tests for the load-optimizer gate (the no-load-optimizer option) of the training
    # state loader and of the adapter loader that forwards it.
    """--no-load-optim must keep the fresh optimizer without losing the step or the LR schedule."""

    @staticmethod
    def _recorder():
        # [FUNCTION DOC] HELPER: Same as the recorder of the previous test class: returns an empty
        # list and a stand-in whose load-state-dict method appends received states to the list.
        loaded = []
        return loaded, SimpleNamespace(load_state_dict=loaded.append)

    @staticmethod
    def _write_training_state(tmp_path):
        # [FUNCTION DOC] HELPER: Writes the rank-0 training-state file with iteration 11, optimizer
        # state of step 7 and scheduler state of lr 0.5.
        torch.save(
            {"iteration": 11, "optimizer": {"step": 7}, "opt_param_scheduler": {"lr": 0.5}},
            tmp_path / "training_state_rank0.pt",
        )

    def test_no_load_optim_skips_the_optimizer_and_keeps_the_rest(self, tmp_path):
        # [FUNCTION DOC] Calls the loader with the load-optimizer flag false. Asserts it returns
        # iteration 11 with optimizer-restored false, the optimizer received nothing and the
        # scheduler received its state.
        self._write_training_state(tmp_path)
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler, load_optimizer=False) == (11, False)
        assert optimizer_loads == []
        assert scheduler_loads == [{"lr": 0.5}]

    def test_load_lora_adapter_forwards_the_flag(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Fakes rank 0 for tensor and pipeline parallelism, writes a matching
        # adapter shard and a training-state file, and loads through the public adapter loader
        # with the load-optimizer flag false. Asserts the result is (loaded true, iteration 11,
        # optimizer restored false), the optimizer received nothing and the scheduler received its
        # state.
        rank0 = SimpleNamespace(rank=0)
        monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: SimpleNamespace(tp=rank0, pp=rank0))
        name = "layers.0.self_attention.lora_A.weight"
        torch.save({name: torch.ones(2)}, tmp_path / "adapter_megatron_rank0.pt")
        self._write_training_state(tmp_path)
        model = [SimpleNamespace(named_parameters=lambda: [(name, torch.nn.Parameter(torch.zeros(2)))])]
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        loaded, iteration, optimizer_restored = load_lora_adapter(
            model,
            str(tmp_path),
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            load_optimizer=False,
        )

        assert (loaded, iteration, optimizer_restored) == (True, 11, False)
        assert optimizer_loads == []
        assert scheduler_loads == [{"lr": 0.5}]
