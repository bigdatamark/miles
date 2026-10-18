# [MODULE DOC]
# CPU unit tests for the backend-neutral Hugging Face weight iterator: placement resolution (joining
# which parallel dimensions must be gathered), the template method that streams base weights followed
# by prefixed adapter weights using a stub iterator subclass with canned units, and the checkpoint
# tower streaming helper (reads vision/audio tower tensors from a small safetensors checkpoint written
# to a temp dir, with or without an index file). Registered for the CPU stage-a suite.
"""Unit tests for the backend-neutral HF weight iterator base.

Covers WeightUpdatePlacement / resolve_placement and the iter_hf_weights
template method (adapter units in the bucketed stream), which every backend
shares.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)

SAMPLE_LORA_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
    ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
]

SAMPLE_BASE_ONLY_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
]


class TestWeightUpdatePlacement:
    # [CLASS DOC] Tests of placement resolution.
    def test_resolve_without_forced_returns_required(self):
        # [FUNCTION DOC] With no forced placement, resolution returns the required placement
        # unchanged.
        required = WeightUpdatePlacement(gather_pp=False)
        assert resolve_placement(required, None) == required

    def test_resolve_joins_gathered_dims(self):
        # [FUNCTION DOC] Resolving a placement that keeps pipeline sharding against one that gathers
        # pipeline stages yields the gathering placement, in either argument order (a join).
        keep_pp = WeightUpdatePlacement(gather_pp=False)
        full = WeightUpdatePlacement(gather_pp=True)
        assert resolve_placement(keep_pp, full) == full
        assert resolve_placement(full, keep_pp) == full


class _StubIterator(HfWeightIteratorBase):
    # [CLASS DOC] HELPER: Concrete iterator subclass that yields canned base units (one single-tensor
    # unit per base pair) and canned adapter units (one single-tensor unit per exported pair), and
    # records each adapter object passed to its adapter hook in the export calls list.
    """Concrete subclass with canned base and adapter unit streams."""

    def __init__(self, exported, base=()):
        # [FUNCTION DOC] HELPER: Initializes the base iterator with a one-gibibyte update buffer, no
        # models, pipeline-gathering placement, model name "stub" and no quantization config, then
        # stores the exported adapter pairs, the base pairs (default empty) and an empty call log.
        super().__init__(
            Namespace(update_weight_buffer_size=1 << 30),
            [],
            placement=WeightUpdatePlacement(gather_pp=True),
            model_name="stub",
            quantization_config=None,
        )
        self._exported = exported
        self._base = base
        self.export_calls = []

    def _iter_hf_param_units(self, weights, *, materialize):
        # [FUNCTION DOC] HELPER: Hook override; ignores its arguments and yields each canned base
        # pair as its own unit.
        for pair in self._base:
            yield [pair]

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        # [FUNCTION DOC] HELPER: Hook override; records the adapter object, then yields each canned
        # exported pair as its own unit.
        self.export_calls.append(adapter)
        for name, tensor in self._exported:
            yield [(name, tensor)]


class TestIterHfWeightsTemplate:
    # [CLASS DOC] Tests of the weight streaming template method using the stub iterator.
    @staticmethod
    def _names(buckets):
        # [FUNCTION DOC] HELPER: Flattens an iterable of buckets (lists of name and tensor pairs)
        # into a single list of tensor names in order.
        return [name for bucket in buckets for name, _ in bucket]

    def test_adapter_units_join_the_stream_prefixed(self):
        # [FUNCTION DOC] Streaming with one adapter named "miles_lora" yields the base tensor name
        # first, then each adapter tensor name prefixed with the adapter name and a colon.
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS, base=SAMPLE_BASE_ONLY_WEIGHTS)
        names = self._names(iterator.iter_hf_weights(None, adapters=[("miles_lora", None)]))
        assert names == [SAMPLE_BASE_ONLY_WEIGHTS[0][0]] + [f"miles_lora:{n}" for n, _ in SAMPLE_LORA_WEIGHTS]

    def test_adapter_argument_reaches_the_hook(self):
        # [FUNCTION DOC] The adapter object paired with the adapter name in the request is the object
        # handed to the adapter hook (the stub's call log holds exactly that object).
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS)
        adapter = SimpleNamespace(slot=3)
        list(iterator.iter_hf_weights(None, adapters=[("__miles_slot_3", adapter)]))
        assert iterator.export_calls == [adapter]

    def test_include_base_false_streams_adapters_only(self):
        # [FUNCTION DOC] With the include-base option false, only prefixed adapter tensor names are
        # streamed.
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS, base=SAMPLE_BASE_ONLY_WEIGHTS)
        names = self._names(iterator.iter_hf_weights(None, include_base=False, adapters=[("miles_lora", None)]))
        assert names == [f"miles_lora:{n}" for n, _ in SAMPLE_LORA_WEIGHTS]

    def test_each_adapter_keeps_its_own_prefix(self):
        # [FUNCTION DOC] Regression test: with two adapters (names containing at-signs), the
        # streamed names are first all tensors prefixed with the first adapter's name, then all
        # prefixed with the second's, so the prefix is bound per adapter and not late-bound.
        """A genexp in the chaining loop late-binds the name and stamps every adapter with the last one."""
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS)
        names = self._names(
            iterator.iter_hf_weights(None, include_base=False, adapters=[("run_a@1", None), ("run_b@2", None)])
        )
        expected = [f"run_a@1:{n}" for n, _ in SAMPLE_LORA_WEIGHTS] + [f"run_b@2:{n}" for n, _ in SAMPLE_LORA_WEIGHTS]
        assert names == expected

    def test_no_adapters_matches_base_stream(self):
        # [FUNCTION DOC] With no adapters requested, only the base tensor name is streamed and the
        # adapter hook is never called.
        iterator = _StubIterator([], base=SAMPLE_BASE_ONLY_WEIGHTS)
        names = self._names(iterator.iter_hf_weights(None))
        assert names == [SAMPLE_BASE_ONLY_WEIGHTS[0][0]]
        assert iterator.export_calls == []


import json

from safetensors.torch import save_file

from miles.backends.training_utils.weight_update.hf_weight_iterator import checkpoint_towers

_TOWERS_MODULE = "miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers"


def _tower_checkpoint(tmp_path, *, indexed: bool):
    # [FUNCTION DOC] HELPER: Writes a single-shard safetensors checkpoint into the directory with
    # four small tensors: a language model layer weight (ones), a vision block norm weight (twos),
    # a vision merger bias (threes) and an audio encoder weight (fours). If indexed, also writes a
    # safetensors index file mapping every tensor to the shard file. Returns the directory path as
    # a string.
    tensors = {
        "model.layers.0.weight": torch.ones(2),
        "visual.blocks.0.norm1.weight": torch.full((3,), 2.0),
        "model.visual.merger.bias": torch.full((1,), 3.0),
        "audio.encoder.weight": torch.full((2,), 4.0),
    }
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}})
        )
    return str(tmp_path)


def _tower_names(units):
    # [FUNCTION DOC] HELPER: Returns the sorted list of tensor names across all units.
    return sorted(name for unit in units for name, _ in unit)


def test_only_tower_keys_stream_with_or_without_an_index(tmp_path):
    # [FUNCTION DOC] For a checkpoint with and without an index file (separate subdirectories, with
    # the CUDA current device patched to report the CPU), streaming tower units with materialization
    # on yields only the audio, vision merger and vision block tensors (the language model layer is
    # excluded), and the vision norm tensor equals the original values (all twos, length three).
    for indexed in (True, False):
        root = tmp_path / ("indexed" if indexed else "flat")
        root.mkdir()
        ckpt = _tower_checkpoint(root, indexed=indexed)
        with patch(f"{_TOWERS_MODULE}.torch.cuda.current_device", return_value="cpu"):
            units = list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=True))
        assert _tower_names(units) == [
            "audio.encoder.weight",
            "model.visual.merger.bias",
            "visual.blocks.0.norm1.weight",
        ]
        assert torch.equal(
            dict(u for unit in units for u in unit)["visual.blocks.0.norm1.weight"], torch.full((3,), 2.0)
        )


def test_non_materializing_ranks_yield_nothing_and_read_nothing(tmp_path):
    # [FUNCTION DOC] With materialization off, tower streaming yields no units and never opens any
    # safetensors file.
    ckpt = _tower_checkpoint(tmp_path, indexed=True)
    with patch(f"{_TOWERS_MODULE}.safe_open") as opened:
        assert list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=False)) == []
    opened.assert_not_called()


def test_the_checkpoint_is_read_once_per_path(tmp_path):
    # [FUNCTION DOC] Clears the module's tower loading cache, streams towers once (reading the
    # files), then streams again for the same path with the file opener patched, and asserts the
    # second pass never opened a file (results are cached per checkpoint path).
    ckpt = _tower_checkpoint(tmp_path, indexed=True)
    checkpoint_towers._load_towers.cache_clear()
    with patch(f"{_TOWERS_MODULE}.torch.cuda.current_device", return_value="cpu"):
        list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=True))
        with patch(f"{_TOWERS_MODULE}.safe_open") as opened:
            list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=True))
    opened.assert_not_called()
