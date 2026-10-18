# [MODULE DOC] CPU test of the direct hugging-face weight iterator's batching helper, which packs
# parameter infos into gather batches by size only (atomicity of related parameters is handled by
# the base template, not here). The iterator module imports heavy dependencies (triton, sglang, ray,
# megatron), so a fixture swaps stub modules into the module table, imports a fresh copy of the
# module, and restores the previous module table entries afterwards. Registered as a CPU test on the
# stage-a-cpu suite.
import sys
import types
from argparse import Namespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.types import ParamInfo


def _install_import_stubs(monkeypatch):
    # [FUNCTION DOC] HELPER: Using the monkeypatch fixture (so everything is undone at test end),
    # registers stub modules in the module table so the iterator module imports without the real
    # packages: triton (with a pass-through jit decorator and a ceiling-division helper) and its
    # language module (constexpr as int); the sglang utility, weight-sync and fp8 quantization
    # modules (serializer and bucket classes as plain object, no-op torch-reduction patch, no-op
    # quantization helpers, scale transform returning its input); ray and its actor and scheduling
    # modules (a remote decorator that returns the decorated object, actor handle and node-affinity
    # strategy as object); and megatron core transformer-layer module with a layer-offset function
    # returning 0.
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.cdiv = lambda x, y: (x + y - 1) // y
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)

    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.utils.patch_torch",
        "sglang.srt.weight_sync",
        "sglang.srt.weight_sync.tensor_bucket",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.fp8_utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    sys.modules["sglang.srt.utils"].MultiprocessingSerializer = object
    sys.modules["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = lambda: None
    sys.modules["sglang.srt.weight_sync.tensor_bucket"].FlattenedTensorBucket = object
    fp8_utils = sys.modules["sglang.srt.layers.quantization.fp8_utils"]
    fp8_utils.quant_weight_ue8m0 = lambda *args, **kwargs: None
    fp8_utils.transform_scale_ue8m0 = lambda x, **kwargs: x

    ray = types.ModuleType("ray")
    ray_actor = types.ModuleType("ray.actor")
    ray_util = types.ModuleType("ray.util")
    ray_scheduling = types.ModuleType("ray.util.scheduling_strategies")
    ray.remote = lambda *args, **kwargs: args[0] if args and callable(args[0]) and not kwargs else lambda obj: obj
    ray_actor.ActorHandle = object
    ray_scheduling.NodeAffinitySchedulingStrategy = object
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", ray_scheduling)

    for name in [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = lambda *args: 0


@pytest.fixture
def direct_module(monkeypatch):
    # [FUNCTION DOC] HELPER: Generator fixture. Saves and removes from the module table seven miles
    # modules (the sglang helper, megatron-to-hf package and its processors and fp8/mxfp8 quantizers,
    # named weights, and the direct iterator) so they are imported fresh; installs the import stubs;
    # imports and yields the direct iterator module. Teardown restores the saved entries (removing
    # ones that were absent).
    module_names = [
        "miles.backends.megatron_utils.sglang",
        "miles.backends.megatron_utils.megatron_to_hf",
        "miles.backends.megatron_utils.megatron_to_hf.processors",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8",
        "miles.backends.megatron_utils.named_weights",
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    _install_import_stubs(monkeypatch)

    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct

    yield hf_weight_iterator_direct

    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _param(name: str, size: int) -> ParamInfo:
    # [FUNCTION DOC] HELPER: Builds a parameter info record with the name, float32 dtype, a
    # one-dimensional shape of the given size, no attributes, the size, and source rank 0.
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([size]),
        attrs={},
        size=size,
        src_rank=0,
    )


def test_gather_batches_pack_by_size_only(direct_module, monkeypatch):
    # [FUNCTION DOC] Three parameters of sizes 4, 2, 4; the module's full-size function is replaced
    # so each parameter's size is its recorded size. With a buffer size of 6 the packing helper must
    # produce batches [a, b] and [c] (greedy by size). With a size multiplier of 2 (each counts double)
    # every parameter is alone: [a], [b], [c].
    # Atomicity is the base template's job; gather batches are pure size packing.
    params = [_param("layer.a", 4), _param("layer.b", 2), _param("layer.c", 4)]
    monkeypatch.setattr(direct_module, "_get_param_full_size", lambda info: info.size)

    batches = direct_module._pack_param_infos_by_size(Namespace(update_weight_buffer_size=6), params)
    assert [[param.name for param in batch] for batch in batches] == [["layer.a", "layer.b"], ["layer.c"]]

    batches = direct_module._pack_param_infos_by_size(
        Namespace(update_weight_buffer_size=6), params, size_multiplier=2
    )
    assert [[param.name for param in batch] for batch in batches] == [["layer.a"], ["layer.b"], ["layer.c"]]
