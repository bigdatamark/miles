# [MODULE DOC]
# Tests of the linear-attention plugin's kernel selection helpers and of numerical equivalence between
# the fla and flashqla gated delta rule kernels. Selection tests run on cpu with mocked modules and
# environment; the equivalence test needs a Hopper or newer cuda gpu with both libraries installed and
# is otherwise skipped. Constants: 4 heads, key and value head dims of 128, and packed sequence lengths
# 128, 256, 128.
"""Linear-attention kernel selection, and fla vs flashqla numerical equivalence for GDN.

The equivalence tests need a Hopper (SM90+) GPU with both `fla` and `flash_qla`; they skip otherwise.
"""

import os
import sys
import types

import pytest

from miles_plugins.models import linear_attn


def test_unknown_backend_raises_value_error():
    # [FUNCTION DOC] Requesting a gated delta rule kernel for an unknown backend name raises a value
    # error saying the backend is unsupported.
    with pytest.raises(ValueError, match="Unsupported GDN backend"):
        linear_attn.gdn_kernel("nope")


@pytest.mark.parametrize(
    "capability,user_value,expected", [((10, 0), None, "0"), ((9, 0), None, None), ((10, 0), "1", "1")]
)
def test_kda_uses_the_triton_backward_on_blackwell_unless_set(monkeypatch, capability, user_value, expected):
    # [FUNCTION DOC] Parametrized over (gpu capability, user value of the tilelang environment variable,
    # expected final value): (10,0) unset gives "0"; (9,0) unset gives unset; (10,0) with user value "1"
    # stays "1". Skips if torch is missing. Fakes the fla kda ops module (chunk kda is a marker string),
    # cuda availability, and capability; clears and optionally sets the variable. Calls the undecorated
    # (unwrapped, bypassing its cache) kda kernel selector, asserts it returns the marker, and that the
    # process environment variable now equals the expected value (none when expected is none).
    # The selector sets the variable as a side effect on blackwell only when the user has not set it.
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "fla.ops.kda", types.SimpleNamespace(chunk_kda="chunk_kda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    if user_value is not None:
        monkeypatch.setenv("FLA_TILELANG", user_value)
    assert linear_attn.kda_kernel.__wrapped__() == "chunk_kda"
    assert os.environ.get("FLA_TILELANG") == expected


def test_short_conv_backend_respects_fla_conv_backend(monkeypatch):
    # [FUNCTION DOC] Sets the fla conv backend environment variable to cuda and asserts the undecorated
    # short convolution backend selector returns cuda.
    monkeypatch.setenv("FLA_CONV_BACKEND", "cuda")
    assert linear_attn.short_conv_backend.__wrapped__() == "cuda"


NUM_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
SEQLENS = [128, 256, 128]


def _require_backends():
    # [FUNCTION DOC] HELPER: Imports torch or skips; skips if no cuda device, if the device major
    # capability is below 9, or if the fla gated delta rule ops or flash qla are not importable. Returns
    # the torch module.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required to compare GDN kernels")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FlashQLA requires NVIDIA SM90 (Hopper) or newer")
    pytest.importorskip("fla.ops.gated_delta_rule")
    pytest.importorskip("flash_qla")
    return torch


def _make_inputs(torch, dtype, device):
    # [FUNCTION DOC] HELPER: Seeds the global generator with 0 and builds random inputs for one packed
    # batch of total length 512: query, key, value of shape 1, total, heads, dim in the given dtype; g, a
    # float32 per-head log decay equal to minus softplus of normals (non-positive); beta, float32
    # sigmoid of normals (in 0 to 1); and int32 cumulative sequence lengths 0, 128, 384, 512. Returns
    # them as a six-tuple in that order.
    torch.manual_seed(0)
    total = sum(SEQLENS)

    def randn(*shape):
        # [FUNCTION DOC] HELPER: Returns standard normal values of the shape in the enclosing dtype and
        # device.
        return torch.randn(*shape, device=device, dtype=dtype)

    query = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    key = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    value = randn(1, total, NUM_HEADS, HEAD_V_DIM)
    # g: per-head log-decay (<= 0); beta: gate in (0, 1) -- as the model feeds them.
    g = -torch.nn.functional.softplus(randn(1, total, NUM_HEADS).float())
    beta = randn(1, total, NUM_HEADS).float().sigmoid()

    cu = torch.tensor([0, *SEQLENS], device=device, dtype=torch.int32).cumsum(0).to(torch.int32)
    return query, key, value, g, beta, cu


def _run(kernel, query, key, value, g, beta, cu_seqlens):
    # [FUNCTION DOC] HELPER: Calls the gated delta rule kernel with contiguous copies of the tensors, no
    # initial state, no final state output, in-kernel l2 normalization of q and k, and the sequence
    # boundaries. Returns only the output (first element of the kernel's result pair).
    out, _ = kernel(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    return out


# FlashQLA only supports half precision (asserts on float32).
@pytest.mark.parametrize(
    "dtype_name, atol, rtol",
    [
        ("bfloat16", 4e-2, 4e-2),
        ("float16", 1e-2, 1e-2),
    ],
)
def test_fla_flashqla_equivalence(dtype_name, atol, rtol):
    # [FUNCTION DOC] Parametrized over bfloat16 (tolerances 0.04) and float16 (0.01); flashqla supports
    # only half precision. Requires backends (else skip), obtains both kernels from the selector, runs
    # each on identical cuda inputs, converts outputs to float32, asserts equal shapes, prints the max
    # and mean absolute difference and max relative difference (relative to the fla maximum plus 1e-6),
    # and asserts closeness within the given absolute and relative tolerances.
    torch = _require_backends()
    dtype = getattr(torch, dtype_name)

    fla_kernel = linear_attn.gdn_kernel("fla")
    flashqla_kernel = linear_attn.gdn_kernel("flashqla")

    query, key, value, g, beta, cu = _make_inputs(torch, dtype, device="cuda")

    fla_out = _run(fla_kernel, query, key, value, g, beta, cu).float()
    flashqla_out = _run(flashqla_kernel, query, key, value, g, beta, cu).float()

    assert fla_out.shape == flashqla_out.shape
    diff = (fla_out - flashqla_out).abs()
    denom = fla_out.abs().max().item() + 1e-6
    print(
        f"\n[GDN fla vs flashqla] dtype={dtype_name} "
        f"max_abs_diff={diff.max().item():.3e} mean_abs_diff={diff.mean().item():.3e} "
        f"max_rel_diff={diff.max().item() / denom:.3e}"
    )

    torch.testing.assert_close(flashqla_out, fla_out, atol=atol, rtol=rtol)
