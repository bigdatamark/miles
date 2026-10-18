# [MODULE DOC]
# Tests for the rematerialization utilities of the Megatron backend, which rebuild low-precision
# model parameters from the optimizer's full-precision master copies: the builder of the
# "cast main params to model params" function for chained distributed optimizers, and the replay
# of hybrid-device optimizer copy-back for the CPU-resident and GPU-resident parameter fractions.
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.rematerialize_utils import (
    _build_cast_main_to_params_fn,
    _replay_hybrid_device_copy_back,
)


class _FakeDistOpt:
    # [CLASS DOC] HELPER: Fake distributed optimizer counting how many times its master-to-model
    # copy was invoked.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts the copy counter at 0.
        self.copied = 0

    def _copy_main_params_to_model_params(self):
        # [FUNCTION DOC] HELPER: Increments the copy counter.
        self.copied += 1


def test_mcore_cast_calls_every_chained_optimizer():
    # [FUNCTION DOC] Builds the cast function over a chain of two fake distributed optimizers
    # without precision-aware mode, calls it, and asserts each fake's copy ran exactly once.
    dist_opts = [_FakeDistOpt(), _FakeDistOpt()]
    cast = _build_cast_main_to_params_fn(SimpleNamespace(chained_optimizers=dist_opts), precision_aware=False)
    cast()
    assert [opt.copied for opt in dist_opts] == [1, 1]


def test_hdo_replay_covers_both_fractions():
    # [FUNCTION DOC] A fake hybrid-device optimizer has a CPU-fraction map (bfloat16 zero view to
    # float32 random master) and a GPU-fraction map (another view to its master). After the
    # replay, asserts each view equals its master cast to bfloat16.
    cpu_view = torch.zeros(4, dtype=torch.bfloat16)
    cpu_master = torch.randn(4)
    gpu_view = torch.zeros(4, dtype=torch.bfloat16)
    gpu_master = torch.randn(4)
    hdo = SimpleNamespace(
        gpu_params_map_cpu_copy={cpu_view: cpu_master},
        param_to_fp32_param={gpu_view: gpu_master},
    )
    _replay_hybrid_device_copy_back(hdo)
    assert torch.equal(cpu_view, cpu_master.to(torch.bfloat16))
    assert torch.equal(gpu_view, gpu_master.to(torch.bfloat16))


def test_hdo_replay_cpu_hook_takes_precedence_on_overlap():
    # [FUNCTION DOC] One view appears in both maps: with master 2.0 in the CPU map and a stale
    # master 3.0 in the GPU map. After the replay, asserts the view equals the CPU master cast to
    # bfloat16 (the GPU pass skips views already handled by the CPU pass).
    # A CPU-fraction param is in both maps; the GPU pass must skip it, like the hooks do.
    view = torch.zeros(4, dtype=torch.bfloat16)
    cpu_master = torch.full((4,), 2.0)
    stale = torch.full((4,), 3.0)
    hdo = SimpleNamespace(
        gpu_params_map_cpu_copy={view: cpu_master},
        param_to_fp32_param={view: stale},
    )
    _replay_hybrid_device_copy_back(hdo)
    assert torch.equal(view, cpu_master.to(torch.bfloat16))


def test_builder_rejects_non_hdo_inner_under_precision_aware():
    # [FUNCTION DOC] Skips if Megatron core is missing. In precision-aware mode, a chain whose
    # member wraps a plain object (not a hybrid-device optimizer) makes the builder raise an
    # assertion error.
    pytest.importorskip("megatron.core")
    optimizer = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=object())])
    with pytest.raises(AssertionError):
        _build_cast_main_to_params_fn(optimizer, precision_aware=True)
