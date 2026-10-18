# [MODULE DOC] CPU tests of the DeepSeek-V3.2 / GLM-5 sparse-attention indexer weight conversion
# between Megatron and Hugging Face layouts, for both the miles implementation (HF-like names, with a
# rope-channel reordering when rope interleave is on) and the native Megatron implementation (names
# under the core-attention indexer). Covers the raw exporter function (names and tensor layout), the
# mbridge bridge classes' export and import round trip (skipped without mbridge), the MLA q/kv
# layernorm export (fused and unfused naming) for both raw and bridge paths, and a numerical test
# (skipped without Megatron's native DSA module) that exporting native weights and applying the HF
# style rotary embedding gives the same projected query/key activations (up to the known channel
# permutation) and the same indexer scores as the native implementation, in float64. Only the converter
# modules are loaded by file path, avoiding unrelated GPU plugins.
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[4]
_INDEXER_WEIGHTS = [
    ("wq_b.weight", "linear_wq_b.weight", (256, 8)),
    ("wk.weight", "linear_wk.weight", (128, 8)),
    ("weights_proj.weight", "linear_weights_proj.weight", (2, 8)),
    ("k_norm.weight", "k_norm.weight", (128,)),
    ("k_norm.bias", "k_norm.bias", (128,)),
]


def _load_module(relative_path, name):
    # [FUNCTION DOC] HELPER: Loads a Python source file given relative to the repository root as a
    # module with the given private name and executes it (without registering it in the module
    # table); returns the module.
    # Load only the converter under test, without unrelated GPU model plugins.
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def raw_converter():
    # [FUNCTION DOC] HELPER: Module-scoped fixture returning the raw DeepSeek-V3 Megatron-to-HF
    # conversion function, loaded from its source file.
    return _load_module(
        "miles/backends/megatron_utils/megatron_to_hf/deepseekv3.py", "dsa_raw_export_under_test"
    ).convert_deepseekv3_to_hf


@pytest.fixture(scope="module")
def bridge_module():
    # [FUNCTION DOC] HELPER: Module-scoped fixture; skips if mbridge is unavailable, otherwise loads
    # and returns the DeepSeek-V3.2 bridge plugin module (containing the V3.2 and GLM bridge classes).
    pytest.importorskip("mbridge")
    return _load_module("miles_plugins/mbridge/deepseek_v32.py", "dsa_mbridge_under_test")


def _weight(shape):
    # [FUNCTION DOC] HELPER: Returns a float32 tensor of the shape filled with consecutive integers
    # starting at 0 in row-major order, so any row or element permutation is detectable.
    return torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)


def _expected_hf_weight(weight, hf_suffix, impl, interleave):
    # [FUNCTION DOC] HELPER: The expected exported tensor. Unchanged for the native implementation,
    # when rope interleave is off, or for the gate-weights projection. Otherwise (miles implementation
    # with interleave) the legacy layout stores the 64 rope channels after the 64 non-rope channels per
    # 128-row indexer head, so rows are permuted: row r of the output is row (r within head + 64 mod
    # 128) of that head.
    if impl == "megatron" or not interleave or hf_suffix == "weights_proj.weight":
        return weight
    # The legacy Miles layout stores the 64 RoPE channels after the 64
    # non-RoPE channels, independently for every indexer query head.
    rows = torch.arange(weight.shape[0])
    hf_rows = rows // 128 * 128 + (rows % 128 + 64) % 128
    return weight[hf_rows]


def _mcore_name(hf_suffix, native_suffix, impl):
    # [FUNCTION DOC] HELPER: Builds the Megatron parameter name for layer 3's self-attention:
    # the native implementation uses the core-attention indexer path with the native suffix; the miles
    # implementation uses the HF-style suffix directly.
    suffix = f"core_attention.indexer.{native_suffix}" if impl == "megatron" else hf_suffix
    return f"decoder.layers.3.self_attention.{suffix}"


@pytest.mark.parametrize("impl", ["miles", "megatron"])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("hf_suffix,native_suffix,shape", _INDEXER_WEIGHTS)
def test_raw_indexer_export_preserves_each_implementation_layout(
    raw_converter, impl, interleave, hf_suffix, native_suffix, shape
):
    # [FUNCTION DOC] Parametrized over both implementations, rope interleave on/off, and the five
    # indexer weights. Converts one weight with the raw exporter given a wrapped-module name and
    # minimal args (hidden 8, 2 heads, 1 query group); asserts exactly one result whose HF name is the
    # layer 3 indexer path with the HF suffix and whose tensor equals the expected layout exactly.
    args = SimpleNamespace(
        hidden_size=8, num_attention_heads=2, num_query_groups=1, indexer_rope_interleave=interleave
    )
    weight = _weight(shape)
    name = "module.module." + _mcore_name(hf_suffix, native_suffix, impl)
    [(hf_name, exported)] = raw_converter(args, name, weight)
    assert hf_name == f"model.layers.3.self_attn.indexer.{hf_suffix}"
    torch.testing.assert_close(exported, _expected_hf_weight(weight, hf_suffix, impl, interleave), rtol=0, atol=0)


@pytest.mark.parametrize(("bridge_class", "interleave"), [("DeepseekV32Bridge", False), ("GlmMoeDsaBridge", True)])
@pytest.mark.parametrize("impl", ["miles", "megatron"])
@pytest.mark.parametrize("hf_suffix,native_suffix,shape", _INDEXER_WEIGHTS)
def test_mbridge_indexer_import_export_matches_raw_layout(
    bridge_module, bridge_class, impl, interleave, hf_suffix, native_suffix, shape
):
    # [FUNCTION DOC] Parametrized over the V3.2 bridge (no interleave) and the GLM bridge
    # (interleave), both implementations and the five weights. Creates the bridge without its
    # constructor with a minimal HF config, model config and vocab setting; exports one weight to HF
    # format and asserts the single HF name and the expected tensor layout; then imports the exported
    # tensor back to Megatron format and asserts it equals the original weight exactly.
    bridge = object.__new__(getattr(bridge_module, bridge_class))
    bridge.hf_config = SimpleNamespace(indexer_rope_interleave=interleave)
    bridge.config = SimpleNamespace(mtp_num_layers=None)
    bridge.make_vocab_size_divisible_by = None
    weight = _weight(shape)
    name = _mcore_name(hf_suffix, native_suffix, impl)
    hf_names, [exported] = bridge._weight_to_hf_format(name, weight)
    assert hf_names == [f"model.layers.3.self_attn.indexer.{hf_suffix}"]
    torch.testing.assert_close(exported, _expected_hf_weight(weight, hf_suffix, impl, interleave), rtol=0, atol=0)
    imported = bridge._weight_to_mcore_format(name, [exported])
    torch.testing.assert_close(imported, weight, rtol=0, atol=0)


@pytest.mark.parametrize("norm", ["q", "kv"])
@pytest.mark.parametrize("layout", ["fused", "unfused"])
def test_mla_norm_raw_export_keeps_hf_order(raw_converter, norm, layout):
    # [FUNCTION DOC] Parametrized over the q and kv norms and the fused (norm weight attached to the
    # up projection) and unfused naming. The raw exporter must map it to the layer 3 HF name for the
    # a-layernorm of that norm with an unchanged 128-element weight.
    suffix = f"linear_{norm}_up_proj.layer_norm_weight" if layout == "fused" else f"{norm}_layernorm.weight"
    name = f"decoder.layers.3.self_attention.{suffix}"
    weight = _weight((128,))
    args = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_query_groups=1)
    [(hf_name, exported)] = raw_converter(args, "module.module." + name, weight)
    assert hf_name == f"model.layers.3.self_attn.{norm}_a_layernorm.weight"
    torch.testing.assert_close(exported, weight, rtol=0, atol=0)


@pytest.mark.parametrize("norm", ["q", "kv"])
@pytest.mark.parametrize("layout", ["fused", "unfused"])
def test_mla_norm_mbridge_round_trip_keeps_hf_order(bridge_module, norm, layout):
    # [FUNCTION DOC] Same parametrization via the V3.2 bridge: export to HF format gives the single
    # a-layernorm name and an unchanged weight, and importing back gives the original weight.
    bridge = object.__new__(bridge_module.DeepseekV32Bridge)
    bridge.hf_config = SimpleNamespace(indexer_rope_interleave=True)
    bridge.config = SimpleNamespace(mtp_num_layers=None)
    bridge.make_vocab_size_divisible_by = None
    suffix = f"linear_{norm}_up_proj.layer_norm_weight" if layout == "fused" else f"{norm}_layernorm.weight"
    name = f"decoder.layers.3.self_attention.{suffix}"
    weight = _weight((128,))
    names, [exported] = bridge._weight_to_hf_format(name, weight)
    assert names == [f"model.layers.3.self_attn.{norm}_a_layernorm.weight"]
    torch.testing.assert_close(exported, weight, rtol=0, atol=0)
    torch.testing.assert_close(bridge._weight_to_mcore_format(name, [exported]), weight, rtol=0, atol=0)


@pytest.fixture(scope="module")
def native_indexer():
    # [FUNCTION DOC] HELPER: Module-scoped fixture; skips if Megatron's DSA module is unavailable,
    # otherwise returns its indexer class (used only for its rope-application method).
    # Optional MCore import; the numerical test itself uses only CPU tensors.
    return pytest.importorskip("megatron.core.transformer.experimental_attention_variant.dsa").DSAIndexer


def _project_indexer(weights, hidden_states, query_latent):
    # [FUNCTION DOC] HELPER: Projects inputs (sequence, batch 1, features) with a dictionary of
    # HF-named indexer weights: queries from the query latent through the query-b projection viewed as
    # (seq, 1, 2 heads, 128); keys from the hidden states through the key projection followed by layer
    # normalization (width 128, epsilon 1e-6, with norm weight and bias) viewed as (seq, 1, 1, 128);
    # gates from the hidden states through the weights projection divided by the square root of
    # 2 x 128. Returns (queries, keys, gates).
    q = torch.nn.functional.linear(query_latent, weights["wq_b.weight"]).view(-1, 1, 2, 128)
    k = torch.nn.functional.linear(hidden_states, weights["wk.weight"])
    k = torch.nn.functional.layer_norm(k, (128,), weights["k_norm.weight"], weights["k_norm.bias"], 1e-6)
    gates = torch.nn.functional.linear(hidden_states, weights["weights_proj.weight"]) / (2 * 128) ** 0.5
    return q, k.view(-1, 1, 1, 128), gates


def _hf_indexer_rope(x, angles, interleave):
    # [FUNCTION DOC] HELPER: Reference rotary embedding on the leading channels (twice the angle
    # count) of the last dimension, leaving the remaining channels unchanged. With interleave the
    # rotated pairs are adjacent channels (even and odd), otherwise the channels are split into two
    # halves; each pair is rotated by the given angles (cos/sin) and written back in the same layout.
    """HF rotates leading channels in adjacent pairs (GLM) or split halves (DSv3.2)."""
    rope_dim = 2 * angles.shape[-1]
    rope, nope = x[..., :rope_dim], x[..., rope_dim:]
    left, right = (rope[..., ::2], rope[..., 1::2]) if interleave else rope.chunk(2, dim=-1)
    first = left * angles.cos() - right * angles.sin()
    second = right * angles.cos() + left * angles.sin()
    rotated = torch.stack((first, second), dim=-1).flatten(-2) if interleave else torch.cat((first, second), -1)
    return torch.cat((rotated, nope), dim=-1)


def _native_rope_in_hf_activation_order(x, rope_dim, interleave):
    # [FUNCTION DOC] HELPER: Returns x unchanged unless interleave is on; then reorders the rope
    # channels from native [all first-half channels, all second-half channels] order into adjacent
    # interleaved pairs, leaving the other channels in place. Q and K receive the same permutation so
    # dot products are unaffected.
    if not interleave:
        return x
    # MCore leaves GLM activations in [even, odd] order after rotating them.
    # Both Q and K have this permutation, so their dot products are unchanged.
    left, right = x[..., :rope_dim].chunk(2, dim=-1)
    return torch.cat((torch.stack((left, right), dim=-1).flatten(-2), x[..., rope_dim:]), dim=-1)


def _indexer_scores(q, k, gates):
    # [FUNCTION DOC] HELPER: Indexer logits: per-head query-key dot products (query position by key
    # position), rectified, multiplied by the gates, and summed over heads. Inputs are (seq, batch,
    # heads, dim) tensors for q and k and (seq, batch, heads) gates; output is (query seq, batch, key seq).
    return (torch.einsum("tbhd,sbhd->tbhs", q, k).relu() * gates.unsqueeze(-1)).sum(dim=2)


@pytest.mark.parametrize("exporter", ["raw", "mbridge"])
@pytest.mark.parametrize("interleave", [False, True], ids=["deepseek-v32", "glm5"])
@pytest.mark.parametrize("rope_dim", [32, 64])
def test_native_indexer_export_preserves_projected_rope_and_scores(
    raw_converter, native_indexer, request, exporter, interleave, rope_dim
):
    # [FUNCTION DOC] Parametrized over the raw or bridge exporter, DeepSeek-V3.2 (split-halves rope)
    # or GLM-5 (interleaved rope) conventions, and rope widths 32 and 64. With seed 713, draws random
    # float64 indexer weights and exports each through the chosen exporter from native Megatron names.
    # Builds random hidden states and query latents (4 positions) and angles from positions
    # 0, 3, 11, 97. Computes native activations by projecting with the native weights and applying the
    # native indexer's rope method (via a simple fake indexer object), and HF activations by projecting
    # with the exported weights and applying the reference HF rope. Asserts, at 1e-12 tolerance, that
    # native q and k (reordered to HF activation order) equal the HF ones, and that the indexer scores
    # of both are equal.
    generator = torch.Generator().manual_seed(713)
    weights = {
        hf_suffix: torch.randn(shape, generator=generator, dtype=torch.float64)
        for hf_suffix, _, shape in _INDEXER_WEIGHTS
    }
    args = SimpleNamespace(
        hidden_size=8, num_attention_heads=2, num_query_groups=1, indexer_rope_interleave=interleave
    )
    if exporter == "mbridge":
        bridge_module = request.getfixturevalue("bridge_module")
        bridge_type = bridge_module.GlmMoeDsaBridge if interleave else bridge_module.DeepseekV32Bridge
        bridge = object.__new__(bridge_type)
        bridge.hf_config = args
        bridge.config = SimpleNamespace(mtp_num_layers=None)
        bridge.make_vocab_size_divisible_by = None
    exported = {}
    for hf_suffix, native_suffix, _ in _INDEXER_WEIGHTS:
        name = _mcore_name(hf_suffix, native_suffix, "megatron")
        weight = weights[hf_suffix]
        if exporter == "raw":
            [(_, exported[hf_suffix])] = raw_converter(args, "module.module." + name, weight)
        else:
            _, [exported[hf_suffix]] = bridge._weight_to_hf_format(name, weight)

    hidden_states = torch.randn(4, 1, 8, generator=generator, dtype=torch.float64)
    query_latent = torch.randn(4, 1, 8, generator=generator, dtype=torch.float64)
    positions = torch.tensor([0, 3, 11, 97], dtype=torch.float64)
    inv_freq = 10000 ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)
    angles = (positions[:, None] * inv_freq).view(4, 1, 1, -1)
    freqs = torch.cat((angles, angles), dim=-1)
    indexer = SimpleNamespace(
        qk_pos_emb_head_dim=rope_dim,
        index_head_dim=128,
        config=SimpleNamespace(
            dsa_indexer_rope_interleaved=interleave,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            mrope_section=None,
        ),
        pg_collection=SimpleNamespace(cp=object()),
    )
    native_q, native_k, native_gates = _project_indexer(weights, hidden_states, query_latent)
    native_q = native_indexer._apply_rope(indexer, native_q, freqs, mscale=1.0)
    native_k = native_indexer._apply_rope(indexer, native_k, freqs, mscale=1.0)
    hf_q, hf_k, hf_gates = _project_indexer(exported, hidden_states, query_latent)
    hf_q = _hf_indexer_rope(hf_q, angles, interleave)
    hf_k = _hf_indexer_rope(hf_k, angles, interleave)
    for native, hf in ((native_q, hf_q), (native_k, hf_k)):
        torch.testing.assert_close(
            _native_rope_in_hf_activation_order(native, rope_dim, interleave), hf, rtol=1e-12, atol=1e-12
        )
    torch.testing.assert_close(
        _indexer_scores(native_q, native_k, native_gates),
        _indexer_scores(hf_q, hf_k, hf_gates),
        rtol=1e-12,
        atol=1e-12,
    )
