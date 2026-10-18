# [MODULE DOC] CPU tests for the FSDP Hf weight iterator (the producer of weight buckets sent to
# the rollout engine). Uses a fake model (config with model type, state dictionary getter, optional
# recorded sync dtypes) and patches the full-parameter gather to an identity function. Verifies that
# every parameter streams under its name, fp32 master weights are cast to the recorded sync dtype,
# non-materializing ranks still join every gather but yield nothing, batched experts are split via
# the registered param transform, and LoRA adapters are refused.
"""FSDP's HF weight iterator."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from miles.backends.fsdp_utils import hf_weight_iterator as iterator_module
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

_ITERATOR_MODULE = "miles.backends.fsdp_utils.hf_weight_iterator"


def _model(state: dict, *, model_type: str = "", sync_dtypes: dict | None = None):
    # [FUNCTION DOC] HELPER: Builds a fake model namespace with a config carrying the model type
    # and a state dictionary getter returning the given mapping; if sync dtypes are given they are
    # attached as the recorded per-parameter sync dtype mapping. Returns it.
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type), state_dict=lambda: state)
    if sync_dtypes is not None:
        model._fsdp_sync_dtypes = sync_dtypes
    return model


def _iterator(model, *, model_name: str = "qwen3forcausallm"):
    # [FUNCTION DOC] HELPER: Builds the FSDP weight iterator for the fake model with a 1 GiB bucket
    # size, a placement requiring pipeline-parallel gathering, the given model name (default a
    # lowercase qwen3 causal LM class name) and no quantization. Returns it.
    return iterator_module.FSDPHfWeightIterator.build(
        Namespace(update_weight_buffer_size=1 << 30),
        model,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name=model_name,
        quantization_config=None,
    )


def _units(iterator, *, materialize=True):
    # [FUNCTION DOC] HELPER: With the full-parameter gather patched to return its input, iterates the
    # iterator's weights (no engine, given materialize flag) and returns a list with one plain
    # dictionary per bucket.
    with patch(f"{_ITERATOR_MODULE}.gather_full_param", side_effect=lambda p, async_op=False: p):
        return [dict(bucket) for bucket in iterator.iter_hf_weights(None, materialize=materialize)]


def test_every_parameter_streams_under_its_hf_name():
    # [FUNCTION DOC] Test: with embedding and lm head entries in the state, the union of names
    # across buckets equals the state's names.
    state = {"model.embed_tokens.weight": torch.zeros(2), "lm_head.weight": torch.ones(2)}
    buckets = _units(_iterator(_model(state)))
    assert {name for bucket in buckets for name in bucket} == set(state)


def test_fp32_master_weights_are_cast_to_the_rollout_contract_dtype():
    # [FUNCTION DOC] Test: two copies of an fp32 value slightly above one, with recorded sync dtypes
    # fp32 and bf16. The fp32-synced tensor stays fp32 and equal to the value; the bf16-synced one
    # is bf16 and, widened back, differs from the value (precision lost).
    value = torch.tensor([1.0 + 2**-20], dtype=torch.float32)
    state = {"fp32_weight": value, "bf16_weight": value.clone()}
    buckets = _units(
        _iterator(_model(state, sync_dtypes={"fp32_weight": torch.float32, "bf16_weight": torch.bfloat16}))
    )
    synced = {name: tensor for bucket in buckets for name, tensor in bucket.items()}
    assert synced["fp32_weight"].dtype is torch.float32 and torch.equal(synced["fp32_weight"], value)
    assert synced["bf16_weight"].dtype is torch.bfloat16
    assert not torch.equal(synced["bf16_weight"].to(torch.float32), value)


def test_non_materializing_ranks_join_every_gather_but_yield_nothing():
    # [FUNCTION DOC] Test: with two parameters and materialization off, iterating yields no buckets
    # yet the gather was called twice (once per parameter), keeping collective participation.
    state = {"a": torch.zeros(1), "b": torch.zeros(1)}
    with patch(f"{_ITERATOR_MODULE}.gather_full_param", side_effect=lambda p, async_op=False: p) as gather:
        buckets = list(_iterator(_model(state)).iter_hf_weights(None, materialize=False))
    assert buckets == []
    assert gather.call_count == 2


def test_batched_experts_are_unfused_through_the_registered_transform():
    # [FUNCTION DOC] Test: a model of type qwen3 moe with one batched fused gate-up parameter of
    # leading size 2; the parameter-transform lookup is patched to return a function yielding one
    # (name, slice) pair per leading index. The streamed names are exactly the two per-expert
    # names, in order.
    """qwen3_moe keeps experts batched as [E, ...]; the engine wants one tensor per expert."""
    gate_up = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 2, 4)
    state = {"model.layers.0.mlp.experts.gate_up_proj": gate_up}
    model = _model(state, model_type="qwen3_moe")
    with (
        patch(f"{_ITERATOR_MODULE}.gather_full_param", side_effect=lambda p, async_op=False: p),
        patch(
            f"{_ITERATOR_MODULE}.get_param_transform",
            return_value=lambda name, full, model: ((f"expert.{i}", full[i]) for i in range(full.shape[0])),
        ),
    ):
        buckets = list(_iterator(model).iter_hf_weights(None))
    names = [name for bucket in buckets for name, _ in bucket]
    assert names == ["expert.0", "expert.1"]


def test_lora_adapters_are_refused():
    # [FUNCTION DOC] Test: asking the iterator's adapter-unit iterator for a lora adapter raises a
    # not-implemented error mentioning LoRA.
    with pytest.raises(NotImplementedError, match="LoRA"):
        list(_iterator(_model({}))._iter_hf_adapter_units("lora", None, materialize=True))
