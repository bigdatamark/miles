# [MODULE DOC] Tests for the MiMo-V2 checkpoint to bf16 conversion tool (tools/convert_mimo_v2_to_bf16).
# Pure CPU tests built on tiny synthetic data. The helpers here fabricate quantized weights (4-bit
# microscaling floats packed two per byte, and 8-bit float weights with 128 by 128 block scales),
# and a miniature two-layer checkpoint on disk in the official fused-QKV layout.
# Covered: exact 4-bit dequantization, per-shard scales of the fused QKV dequantization, splitting
# fused QKV back into separate projections, an end-to-end conversion (with and without layer and
# expert subsetting), re-indexing of ignored layers when quantization is kept, and atomicity of
# the output index when copying auxiliary files fails.
import json

import pytest
import tools.convert_mimo_v2_to_bf16 as converter
import torch
from safetensors.torch import load_file, save_file
from tools.convert_mimo_v2_to_bf16 import FP4_TABLE, dequant_fused_qkv, dequant_mxfp4, main, split_fused_qkv

E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _pack_mxfp4(codes: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Pack 4-bit codes into bytes. Input is a two-dimensional integer tensor
    # (rows by columns, even column count) of indices into the 4-bit float value table. Output is
    # an unsigned 8-bit tensor with half as many columns, where the even-position element of each
    # pair occupies the low nibble and the odd-position element the high nibble.
    """codes: [out, in] 4-bit indices into FP4_TABLE -> [out, in / 2] uint8, element 2i in the low nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def _quant_fp8_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Quantize one float matrix to 8-bit float (e4m3) with 128 by 128 block
    # scales. The matrix is zero-padded up to whole multiples of 128 in both dimensions, split into
    # blocks, and each block gets a scale equal to its maximum absolute value (floored at a tiny
    # positive number to avoid dividing by zero) divided by the largest representable e4m3 value.
    # Returns (quantized matrix cropped back to the original shape, scale grid with one entry per
    # block, shaped rows-blocks by column-blocks). Scales are multiplicative: weight is approximately
    # quantized value times scale.
    """(128, 128) block quantization of one matrix, rows padded to whole blocks."""
    rows, cols = weight.shape
    padded = torch.zeros(-(-rows // 128) * 128, -(-cols // 128) * 128)
    padded[:rows, :cols] = weight
    blocks = padded.view(padded.shape[0] // 128, 128, padded.shape[1] // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / E4M3_MAX
    quant = (blocks / scale[:, None, :, None]).view_as(padded)[:rows, :cols].to(torch.float8_e4m3fn)
    return quant, scale


def _interleave_qkv(q, k, v, shards):
    # [FUNCTION DOC] HELPER: Build a fused QKV matrix laid out per tensor-parallel shard.
    # Splits each of the query, key and value matrices into the given number of equal chunks along
    # the first dimension, then for each shard concatenates that shard's query, key and value chunks
    # (in that order), and finally concatenates all shards. Returns the fused matrix.
    return torch.cat(
        [torch.cat(parts) for parts in zip(q.chunk(shards), k.chunk(shards), v.chunk(shards), strict=True)]
    )


def test_dequant_mxfp4_is_exact():
    # [FUNCTION DOC] Generates random 4-bit codes (8 rows, 64 columns) and random biased exponent
    # bytes (in 120 to 133, one per 32-element group, so 2 per row). The expected result is the table
    # value times two to the power of (exponent minus 127), each exponent repeated across its 32
    # columns. Asserts the tool's 4-bit dequantization of the packed data returns exactly the
    # expected values after rounding to bfloat16.
    codes = torch.randint(0, 16, (8, 64))
    exponents = torch.randint(120, 134, (8, 2), dtype=torch.uint8)
    expected = FP4_TABLE[codes] * torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=1)
    assert torch.equal(dequant_mxfp4(_pack_mxfp4(codes), exponents).float(), expected.to(torch.bfloat16).float())


def test_fused_qkv_scales_are_per_shard():
    # [FUNCTION DOC] Builds a random fused matrix of two shards of 224 rows by 256 columns (224 is
    # not a multiple of 128) and quantizes each shard separately, so each shard has its own two
    # scale rows (scale grid shape asserted as 4 by 2). The expected result expands each shard's
    # scales to element resolution, crops to 224 rows, and multiplies by the quantized values.
    # Asserts the tool's fused dequantization equals that exactly in bfloat16, and that the
    # reconstruction error against the original float weight stays under 7 percent of the largest
    # absolute weight.
    # Shards of 224 rows are not 128-aligned: each one carries its own 2 scale rows.
    shards, rows = 2, 224
    weight = torch.randn(shards * rows, 256)
    quant_scale = [_quant_fp8_block(part) for part in weight.chunk(shards)]
    fp8 = torch.cat([q for q, _ in quant_scale])
    scale = torch.cat([s for _, s in quant_scale])
    assert scale.shape == (shards * 2, 2)
    expected = torch.cat(
        [q.float() * s.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1) for q, s in quant_scale]
    )
    out = dequant_fused_qkv(fp8, scale, shards)
    assert torch.equal(out, expected.to(torch.bfloat16))
    assert (out.float() - weight).abs().max() < 0.07 * weight.abs().max()


def test_split_fused_qkv_restores_head_order():
    # [FUNCTION DOC] Uses a small config (2 key-value heads, sliding-window attention with 8 query
    # heads, 4 key-value heads, key head size 3 and value head size 2) and random query, key and
    # value matrices with 5 columns. Fuses them with two shards, splits with the sliding-window flag
    # set, and asserts the three outputs are exactly equal to the original query, key and value
    # matrices, i.e. the sliding-window head counts and sizes are used and shard interleaving is undone.
    config = {
        "num_key_value_heads": 2,
        "swa_num_attention_heads": 8,
        "swa_num_key_value_heads": 4,
        "swa_head_dim": 3,
        "swa_v_head_dim": 2,
    }
    q, k, v = torch.randn(8 * 3, 5), torch.randn(4 * 3, 5), torch.randn(4 * 2, 5)
    out = split_fused_qkv(_interleave_qkv(q, k, v, shards=2), config, is_swa=True)
    for got, want in zip(out, (q, k, v), strict=True):
        assert torch.equal(got, want)


def _tiny_checkpoint(path):
    # [FUNCTION DOC] HELPER: Write a miniature MiMo-V2 checkpoint into an existing directory.
    # Layer 0 is global attention with a dense MLP; layer 1 is sliding-window attention with a
    # mixture-of-experts MLP (4 routed experts, 2 per token). Hidden size is 128, 4 query heads,
    # key head size 64, value head size 32, and the config declares the fused-QKV layout and a
    # quantization config whose ignored layers list the layer-1 output projection.
    # Tensors written (single safetensors shard): bf16 embeddings, per layer an FP8 block-quantized
    # fused QKV weight (two shards, so scales are per shard) with inverse-scale tensor, layer norms
    # of ones, a bf16 output projection, a zero attention-sink bias for layer 1, FP8 block-quantized
    # dense gate/up/down projections for layer 0, a bf16 router gate and a float32 score-correction
    # bias for layer 1, and per expert packed 4-bit gate/up/down weights with exponent scales
    # (random, exponents 124 to 129). Also writes the weight-map index file, the config file, and
    # an empty tokenizer config file.
    # Returns a mapping from output tensor name to the expected bf16 result: separate query, key,
    # value projection weights per layer (from dequantizing and splitting via the tool's own helpers)
    # and the dequantized expert weights. Uses unseeded random data.
    """Two layers in the official MiMo-V2 format: global+dense (0) and SWA+MoE (1)."""
    config = {
        "architectures": ["MiMoV2ForCausalLM"],
        "attention_projection_layout": "fused_qkv",
        "num_hidden_layers": 2,
        "hybrid_layer_pattern": [0, 1],
        "moe_layer_freq": [0, 1],
        "hybrid_block_size": None,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "v_head_dim": 32,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 4,
        "swa_head_dim": 64,
        "swa_v_head_dim": 32,
        "add_swa_attention_sink_bias": True,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "quantization_config": {
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "ignored_layers": ["model.layers.1.self_attn.o_proj"],
        },
    }
    expected, tensors = {}, {"model.embed_tokens.weight": torch.randn(16, 128).bfloat16()}
    for layer, (kv_heads, is_swa) in enumerate(((2, False), (4, True))):
        p = f"model.layers.{layer}."
        q, k, v = torch.randn(4 * 64, 128), torch.randn(kv_heads * 64, 128), torch.randn(kv_heads * 32, 128)
        fused = _interleave_qkv(q, k, v, shards=2)
        quant_scale = [_quant_fp8_block(part) for part in fused.chunk(2)]
        tensors[p + "self_attn.qkv_proj.weight"] = torch.cat([qq for qq, _ in quant_scale])
        tensors[p + "self_attn.qkv_proj.weight_scale_inv"] = torch.cat([s for _, s in quant_scale])
        deq = dequant_fused_qkv(
            tensors[p + "self_attn.qkv_proj.weight"], tensors[p + "self_attn.qkv_proj.weight_scale_inv"], 2
        )
        for proj, part in zip("qkv", split_fused_qkv(deq, config, is_swa), strict=True):
            expected[p + f"self_attn.{proj}_proj.weight"] = part
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            tensors[p + name] = torch.ones(128).bfloat16()
        tensors[p + "self_attn.o_proj.weight"] = torch.randn(128, 4 * 32).bfloat16()
    tensors["model.layers.1.self_attn.attention_sink_bias"] = torch.zeros(4).bfloat16()
    for proj, shape in (("gate_proj", (256, 128)), ("up_proj", (256, 128)), ("down_proj", (128, 256))):
        quant, scale = _quant_fp8_block(torch.randn(shape))
        tensors[f"model.layers.0.mlp.{proj}.weight"] = quant
        tensors[f"model.layers.0.mlp.{proj}.weight_scale_inv"] = scale
    tensors["model.layers.1.mlp.gate.weight"] = torch.randn(4, 128).bfloat16()
    tensors["model.layers.1.mlp.gate.e_score_correction_bias"] = torch.randn(4)
    for expert in range(4):
        for proj, (rows, cols) in (("gate_proj", (64, 128)), ("up_proj", (64, 128)), ("down_proj", (128, 64))):
            name = f"model.layers.1.mlp.experts.{expert}.{proj}."
            codes = torch.randint(0, 16, (rows, cols))
            exps = torch.randint(124, 130, (rows, cols // 32), dtype=torch.uint8)
            tensors[name + "weight"], tensors[name + "weight_scale"] = _pack_mxfp4(codes), exps
            expected[name + "weight"] = dequant_mxfp4(tensors[name + "weight"], exps)
    save_file(tensors, str(path / "model-00001.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model-00001.safetensors")})
    )
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text("{}")
    return expected


@pytest.mark.parametrize("layers, num_experts", [(None, None), ([1], 2)])
def test_convert_end_to_end(tmp_path, layers, num_experts):
    # [FUNCTION DOC] Parametrized over (all layers, all experts) and (only layer 1, 2 experts).
    # Creates the tiny checkpoint, runs the tool's main on CPU with quantization dropped, then loads
    # the output config, index and tensors. Asserts: the quantization config and fused-layout marker
    # are gone from the config; the tokenizer config was copied; no scale tensors or fused QKV
    # weights remain; the layer count equals the number of kept layers and the layer pattern and
    # expert-frequency lists equal the kept old layer indices; every expected tensor of a kept layer
    # (and kept expert) is present under its renumbered layer index and bitwise equal; exactly three
    # expert tensors per kept expert exist; the routed-expert count in the config equals the kept
    # experts; the router gate has one row per kept expert; and the correction bias is float32.
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    expected = _tiny_checkpoint(src)
    main(str(src), str(dst), layers, num_experts, keep_quant=False, device="cpu")

    config = json.loads((dst / "config.json").read_text())
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    out = {name: t for f in set(index.values()) for name, t in load_file(str(dst / f)).items()}
    assert "quantization_config" not in config and "attention_projection_layout" not in config
    assert (dst / "tokenizer_config.json").exists()
    assert not any(name.endswith(("weight_scale", "weight_scale_inv", "qkv_proj.weight")) for name in out)
    remap = {old: new for new, old in enumerate(layers)} if layers else {0: 0, 1: 1}
    assert config["num_hidden_layers"] == len(remap)
    assert config["hybrid_layer_pattern"] == config["moe_layer_freq"] == list(remap)
    num_kept_experts = num_experts or 4
    kept = {}
    for name, want in expected.items():
        _, _, old, *rest = name.split(".")
        if int(old) in remap and not (rest[1] == "experts" and int(rest[2]) >= num_kept_experts):
            kept[f"model.layers.{remap[int(old)]}.{'.'.join(rest)}"] = want
    assert len(kept) == (3 if 0 in remap else 0) + 3 + 3 * num_kept_experts
    for name, want in kept.items():
        assert torch.equal(out[name], want), name
    experts = {name for name in out if ".experts." in name}
    assert len(experts) == 3 * num_kept_experts
    assert config["n_routed_experts"] == num_kept_experts
    assert out[f"model.layers.{remap[1]}.mlp.gate.weight"].shape[0] == num_kept_experts
    assert out[f"model.layers.{remap[1]}.mlp.gate.e_score_correction_bias"].dtype == torch.float32


def test_keep_quant_reindexes_ignored_layers(tmp_path):
    # [FUNCTION DOC] Converts the tiny checkpoint keeping only layer 1 with the keep-quantization
    # option on. Asserts the output config's ignored-layers list names the renumbered layer 0 output
    # projection instead of layer 1, and that the fused-QKV layout marker is still present.
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _tiny_checkpoint(src)
    main(str(src), str(dst), [1], None, keep_quant=True, device="cpu")
    config = json.loads((dst / "config.json").read_text())
    assert config["quantization_config"]["ignored_layers"] == ["model.layers.0.self_attn.o_proj"]
    assert config["attention_projection_layout"] == "fused_qkv"


def test_conversion_is_not_complete_until_auxiliary_files_are_copied(tmp_path, monkeypatch):
    # [FUNCTION DOC] Verifies the output index file is written only after auxiliary files are
    # copied. First patches the tool's file-copy function to raise an operating-system error and
    # asserts main propagates that error and leaves no index file in the output directory. Then
    # restores the real copy function, reruns into the same output directory, and asserts the index
    # now has a non-empty weight map and the tokenizer config is a faithful copy of the source one.
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _tiny_checkpoint(src)
    copy2 = converter.shutil.copy2

    def interrupted_copy(source, destination):
        # [FUNCTION DOC] HELPER: Replacement for the file-copy function that ignores its arguments
        # and always raises an operating-system error saying the auxiliary copy was interrupted.
        raise OSError("interrupted auxiliary copy")

    monkeypatch.setattr(converter.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="interrupted auxiliary copy"):
        main(str(src), str(dst), [1], None, keep_quant=True, device="cpu")
    assert not (dst / "model.safetensors.index.json").exists()

    monkeypatch.setattr(converter.shutil, "copy2", copy2)
    main(str(src), str(dst), [1], None, keep_quant=True, device="cpu")
    assert json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert (dst / "tokenizer_config.json").read_text() == (src / "tokenizer_config.json").read_text()
