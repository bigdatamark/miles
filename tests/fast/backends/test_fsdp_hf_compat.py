# [MODULE DOC]
# CPU-only tests for the experimental FSDP HuggingFace-compatibility adaptations: unfusing batched
# mixture-of-experts parameters into per-expert names for weight sync (gate/up row split, down
# projection, contiguity, model-type gating, pass-through streaming when no transform applies);
# gating and lazy loading of the Nemotron-H post-load fixup and its reload of clobbered mixer
# weights; the parameter-transform registry; the model-patch hook registry and the model-type
# verification hook (which warns once on rank zero for unverified types); packed-sequence context
# derivation and the packing patch registry; Nemotron-H attention reusing a precomputed maximum
# sequence length; repair of a broken pattern-to-list method patched by SGLang's import; and
# routing of model configs to the right HuggingFace auto-model class in the FSDP actor. Small
# randomly initialized models are built from tiny configs; some tests mutate library classes and
# restore them in finally blocks.
"""Unit tests for the experimental-FSDP HF-compat fixes (CPU-only, no GPU/sglang).

Covers:
  * F9 weight-sync: batched MoE expert params are unfused through transformers' own
    reverse conversion (``revert_weight_conversion``) into the per-expert names
    SGLang expects, with the correct gate/up row split, contiguous tensors, and
    only for the right model types. These tests pin the HF revert output to the
    on-disk dialect; any upstream drift in the qwen2_moe conversion family or in
    per-tensor revert semantics turns them red.
"""

import logging
import subprocess
import sys

import pytest
import torch

from miles.backends.fsdp_utils import hf_weight_iterator
from miles.backends.fsdp_utils.adaptations.weight_bridge import _hf_unfuse_experts_expand, get_param_transform


@pytest.fixture(scope="module")
def tiny_qwen3_moe():
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Builds and returns a randomly initialized tiny
    # Qwen3 mixture-of-experts causal language model (hidden 16, 1 layer, 4 heads, 2 KV heads, 2
    # experts, 2 experts per token, vocabulary 64, sparse step 1, head dimension 4).
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=2,
        vocab_size=64,
        decoder_sparse_step=1,
        head_dim=4,
    )
    return Qwen3MoeForCausalLM(cfg)


def test_unfuse_gate_up_proj_rows_and_names(tiny_qwen3_moe):
    # [FUNCTION DOC] Unfuses a fused gate/up expert tensor of shape (2 experts, 6 rows, 4) filled
    # with a ramp for layer 0. Asserts the output names are the per-expert gate and up projection
    # weight names for experts 0 and 1; each gate is the first 3 rows and each up the last 3 rows
    # of its expert, shaped (3, 4), and contiguous.
    # [E=2, 2*inter=6, H=4]: fused rows are [gate(:3) | up(3:)]
    E, inter, H = 2, 3, 4
    full = torch.arange(E * 2 * inter * H, dtype=torch.float32).reshape(E, 2 * inter, H)
    out = dict(_hf_unfuse_experts_expand("model.layers.0.mlp.experts.gate_up_proj", full, tiny_qwen3_moe))

    assert set(out) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    }
    for e in range(E):
        g = out[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
        u = out[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
        assert g.shape == (inter, H) and u.shape == (inter, H)
        torch.testing.assert_close(g, full[e, :inter, :])
        torch.testing.assert_close(u, full[e, inter:, :])
        assert g.is_contiguous() and u.is_contiguous()


def test_unfuse_down_proj(tiny_qwen3_moe):
    # [FUNCTION DOC] Unfuses a down-projection expert tensor of shape (2, 4, 3) for layer 5.
    # Asserts the per-expert down projection weight names for experts 0 and 1; each equals its
    # expert's slice, shaped (4, 3), and contiguous.
    E, H, inter = 2, 4, 3
    full = torch.arange(E * H * inter, dtype=torch.float32).reshape(E, H, inter)
    out = dict(_hf_unfuse_experts_expand("model.layers.5.mlp.experts.down_proj", full, tiny_qwen3_moe))
    assert set(out) == {
        "model.layers.5.mlp.experts.0.down_proj.weight",
        "model.layers.5.mlp.experts.1.down_proj.weight",
    }
    for e in range(E):
        d = out[f"model.layers.5.mlp.experts.{e}.down_proj.weight"]
        assert d.shape == (H, inter)
        torch.testing.assert_close(d, full[e])
        assert d.is_contiguous()


def test_unfuse_glm4_moe_lite_same_family():
    # [FUNCTION DOC] Builds a tiny GLM4 MoE-lite model (2 layers, 2 routed experts, one shared
    # expert, first layer dense) and unfuses its real layer-1 fused gate/up parameter. Asserts the
    # per-expert gate and up names for each expert, with gate equal to the first half and up the
    # second half of the rows of each expert's slice.
    # glm4_moe_lite shares qwen3_moe's conversion family (qwen2_moe); its real batched
    # params must unfuse to the same per-expert dialect.
    from transformers import Glm4MoeLiteConfig
    from transformers.models.glm4_moe_lite import Glm4MoeLiteForCausalLM

    cfg = Glm4MoeLiteConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=2,
        vocab_size=64,
        first_k_dense_replace=1,
        n_shared_experts=1,
        head_dim=4,
    )
    model = Glm4MoeLiteForCausalLM(cfg)
    name = "model.layers.1.mlp.experts.gate_up_proj"
    full = model.state_dict()[name]
    E, two_inter = full.shape[0], full.shape[1]
    out = dict(_hf_unfuse_experts_expand(name, full, model))
    assert set(out) == {
        f"model.layers.1.mlp.experts.{e}.{proj}.weight" for e in range(E) for proj in ("gate_proj", "up_proj")
    }
    for e in range(E):
        torch.testing.assert_close(
            out[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"], full[e, : two_inter // 2, :]
        )
        torch.testing.assert_close(out[f"model.layers.1.mlp.experts.{e}.up_proj.weight"], full[e, two_inter // 2 :, :])


def test_param_transform_gating():
    # [FUNCTION DOC] Asserts the parameter-transform lookup returns a transform only for a
    # three-dimensional fused expert parameter of the qwen3_moe type: none for qwen3_5_moe (consumes
    # batched weights directly), none for dense qwen3, none for non-expert parameters, and none
    # for a two-dimensional tensor named like an expert parameter.
    def applies(name, param, model_type):
        # [FUNCTION DOC] HELPER: True when the transform lookup returns something for the arguments.
        return get_param_transform(name, param, model_type) is not None

    gate_up = torch.zeros(2, 6, 4)
    name = "model.layers.0.mlp.experts.gate_up_proj"
    # only for model types whose SGLang loader expects per-expert weights
    assert applies(name, gate_up, "qwen3_moe")
    assert not applies(name, gate_up, "qwen3_5_moe")  # consumes batched directly
    assert not applies(name, gate_up, "qwen3")  # dense
    # non-expert params are never split
    assert not applies("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4), "qwen3_moe")
    # 2D tensor named like an expert param is not the batched layout
    assert not applies(name, torch.zeros(6, 4), "qwen3_moe")


def test_the_iterator_streams_params_without_a_transform_unchanged():
    # [FUNCTION DOC] Creates the FSDP HF weight iterator without its constructor, with a fake
    # model of type qwen3_5_moe whose state dictionary has an embedding weight and a fused expert
    # tensor, a 1 GiB update buffer size and empty sync dtypes; the full-parameter gather is
    # patched to the identity. Lists the iterator's units with materialization on. Asserts two
    # units, each a single named parameter in state-dictionary order, with the very same tensor
    # objects (no transform applied).
    from types import SimpleNamespace
    from unittest.mock import patch

    embed = torch.zeros(4, 4)
    experts = torch.zeros(2, 6, 4)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe"),
        state_dict=lambda: {"model.embed_tokens.weight": embed, "model.layers.0.mlp.experts.gate_up_proj": experts},
    )
    iterator = object.__new__(hf_weight_iterator.FSDPHfWeightIterator)
    iterator.model = model
    iterator.args = SimpleNamespace(update_weight_buffer_size=1 << 30)
    iterator._sync_dtypes = {}
    with patch.object(hf_weight_iterator, "gather_full_param", lambda t, async_op=False: t):
        units = list(iterator._iter_hf_param_units(None, materialize=True))
    assert [[name for name, _ in unit] for unit in units] == [
        ["model.embed_tokens.weight"],
        ["model.layers.0.mlp.experts.gate_up_proj"],
    ]
    assert units[0][0][1] is embed and units[1][0][1] is experts


def test_nemotron_h_post_load_fixup_gating():
    # [FUNCTION DOC] Looks up the nemotron_h_clobber_reload fixup in the fixup registry by name and
    # asserts it applies to the nemotron_h model type only (not mamba2, not a hybrid config with
    # mamba and attention layer types, not qwen3_moe).
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.post_load_fixups import _FIXUPS

    by_name = {f.name: f for f in _FIXUPS}
    fixup = by_name["nemotron_h_clobber_reload"]
    assert fixup.applies_to(SimpleNamespace(model_type="nemotron_h"))
    assert not fixup.applies_to(SimpleNamespace(model_type="mamba2"))
    assert not fixup.applies_to(SimpleNamespace(model_type="hybrid", layer_types=["mamba", "attention"]))
    assert not fixup.applies_to(SimpleNamespace(model_type="qwen3_moe"))


def test_post_load_fixup_lazily_loads_model_implementation():
    # [FUNCTION DOC] Runs a script in a fresh Python subprocess (so module-table state is clean),
    # capturing output. The script asserts exactly one Nemotron-H fixup is registered and that the
    # Nemotron-H model implementation module is not loaded; applying the post-load fixups to a
    # mamba2 config leaves it not loaded; applying them to a nemotron_h config with an empty
    # temporary checkpoint directory loads it. Asserts the subprocess exit code is 0, with its
    # standard error as the failure message.
    script = """
import sys
import tempfile
from types import SimpleNamespace

from miles.backends.fsdp_utils.adaptations.post_load_fixups import (
    _FIXUPS,
    apply_post_load_fixups,
)

module_name = "miles.backends.fsdp_utils.models.nemotron_h"
assert [fixup.name for fixup in _FIXUPS].count("nemotron_h_clobber_reload") == 1
assert module_name not in sys.modules
apply_post_load_fixups(object(), SimpleNamespace(model_type="mamba2"), ".")
assert module_name not in sys.modules
with tempfile.TemporaryDirectory() as ckpt_path:
    apply_post_load_fixups(object(), SimpleNamespace(model_type="nemotron_h"), ckpt_path)
assert module_name in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reload_nemotron_h_clobbered_weights_behavior(tmp_path, caplog):
    # [FUNCTION DOC] Builds a one-layer fake model with mixer parameters: a time-step bias of
    # 1.0005 and output/input projection weights filled with 2.0 and 3.0. With an empty checkpoint
    # directory the reload returns 0. Then saves a checkpoint whose bias is 1.0, output projection
    # 1.0 and input projection 1.0. Reload with default tolerance returns 1 (only the bias differs
    # slightly, and is kept at 1.0005; output projection weight becomes 1.0; the input projection
    # stays 3.0, i.e. only parameters that look clobbered are restored). Reload with tolerance
    # 1e-4 and info-level logging returns 1 and sets the bias to 1.0, leaves the input projection
    # at 3.0, and logs "restored 1 NemotronH mixer parameter(s)" from the module's logger.
    from safetensors.torch import save_file

    from miles.backends.fsdp_utils.models.nemotron_h import reload_nemotron_h_clobbered_weights

    class Mixer(torch.nn.Module):
        # [CLASS DOC] HELPER: Fake mixer with a time-step bias parameter and two bias-free 1-by-1
        # linear layers (output and input projections).
        def __init__(self):
            # [FUNCTION DOC] HELPER: Creates the bias (1.0005) and the projections; fills the
            # output projection weight with 2.0 and the input projection weight with 3.0.
            super().__init__()
            self.dt_bias = torch.nn.Parameter(torch.tensor([1.0005]))
            self.out_proj = torch.nn.Linear(1, 1, bias=False)
            self.in_proj = torch.nn.Linear(1, 1, bias=False)
            self.out_proj.weight.data.fill_(2.0)
            self.in_proj.weight.data.fill_(3.0)

    class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Fake model with a backbone list holding one layer that has a mixer.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Creates the backbone with a single layer module whose mixer
            # attribute is a fake mixer.
            super().__init__()
            layer = torch.nn.Module()
            layer.mixer = Mixer()
            self.backbone = torch.nn.ModuleList([layer])

    model = Model()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert reload_nemotron_h_clobbered_weights(model, empty) == 0

    disk = {
        "backbone.0.mixer.dt_bias": torch.ones(1),
        "backbone.0.mixer.out_proj.weight": torch.ones(1, 1),
        "backbone.0.mixer.in_proj.weight": torch.ones(1, 1),
    }
    save_file(disk, tmp_path / "model.safetensors")
    assert reload_nemotron_h_clobbered_weights(model, tmp_path) == 1
    torch.testing.assert_close(model.backbone[0].mixer.dt_bias, torch.tensor([1.0005]))
    torch.testing.assert_close(model.backbone[0].mixer.out_proj.weight, torch.ones(1, 1))
    torch.testing.assert_close(model.backbone[0].mixer.in_proj.weight, torch.tensor([[3.0]]))

    with caplog.at_level(
        logging.INFO,
        logger="miles.backends.fsdp_utils.models.nemotron_h",
    ):
        assert reload_nemotron_h_clobbered_weights(model, tmp_path, tol=1e-4) == 1
    torch.testing.assert_close(model.backbone[0].mixer.dt_bias, torch.ones(1))
    torch.testing.assert_close(model.backbone[0].mixer.in_proj.weight, torch.tensor([[3.0]]))
    assert any(
        record.name == "miles.backends.fsdp_utils.models.nemotron_h"
        and "restored 1 NemotronH mixer parameter(s)" in record.getMessage()
        for record in caplog.records
    )


def test_weight_bridge_registry():
    # [FUNCTION DOC] Asserts a transform exists for the fused expert parameter of type qwen3_moe
    # and none for qwen3_5_moe. Registers a new transform for a test architecture (matching names
    # ending in ".foo", expanding to the name with ".bar") and asserts the lookup for "x.foo"
    # returns a function whose first output name is "x.bar", and the lookup for "x.baz" returns none.
    # KNOWN ISSUE: the registration of the test architecture is never undone, so it stays in the
    # global registry for the rest of the process.
    # The WeightBridge registry is the train->rollout param-name/shape contract: a model type with
    # a registered transform gets its params rewritten; unregistered types stream verbatim.
    import torch

    from miles.backends.fsdp_utils.adaptations.weight_bridge import get_param_transform, register_param_transform

    # qwen3_moe is registered (batched experts -> per-expert); a 3D experts param matches.
    g = torch.zeros(2, 6, 4)
    assert get_param_transform("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_moe") is not None
    # unregistered model type -> no transform (passthrough)
    assert get_param_transform("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_5_moe") is None
    # registering a new transform routes matching params through it
    register_param_transform(
        "_test_arch",
        matches=lambda name, p: name.endswith(".foo"),
        expand=lambda name, full: [(name.replace(".foo", ".bar"), full)],
    )
    fn = get_param_transform("x.foo", g, "_test_arch")
    assert fn is not None and list(fn("x.foo", g))[0][0] == "x.bar"
    assert get_param_transform("x.baz", g, "_test_arch") is None


def test_model_patch_registry_gating():
    # [FUNCTION DOC] Asserts the first three registered model-patch hooks are, in order,
    # fp8_checkpoint_guard, dsa_train_infer_warn and model_type_verified; the verified hook appears
    # exactly once; no GDN packing hook is registered; the fp8 guard does not apply to a missing
    # config; the qwen3_moe MoE-block hook is registered, applies to the qwen3_moe model type and
    # not to qwen3; and applying that hook with true-on-policy off completes without error.
    # The ModelPatchHook registry replaces the hardcoded per-arch dispatch in apply_class_patches.
    # Verify the config-check predicates gate correctly. Packed-sequence layout patches (GDN, ...) moved
    # out of this registry into the unified packing registry (test_packing_registry below);
    # apply_class_patches now dispatches them via apply_packing.
    from miles.backends.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS

    by_name = {h.name: h for h in _MODEL_PATCH_HOOKS}
    # the expected generic hooks are registered in order (GDN packing is not a ModelPatchHook)
    assert [h.name for h in _MODEL_PATCH_HOOKS][:3] == [
        "fp8_checkpoint_guard",
        "dsa_train_infer_warn",
        "model_type_verified",
    ]
    assert [h.name for h in _MODEL_PATCH_HOOKS].count("model_type_verified") == 1
    assert "gated_deltanet_packing" not in by_name
    assert not by_name["fp8_checkpoint_guard"].applies_to(None)
    # the qwen3_moe MoE-block patch is a hook now (moved out of _enable_true_on_policy_optimizations),
    # gated on model_type; the backend-level enable_batch_invariant_mode stays in the actor.
    from types import SimpleNamespace

    assert "qwen3_moe_moe_patch" in by_name
    assert by_name["qwen3_moe_moe_patch"].applies_to(SimpleNamespace(model_type="qwen3_moe"))
    assert not by_name["qwen3_moe_moe_patch"].applies_to(SimpleNamespace(model_type="qwen3"))
    # Batched experts need no off-mode patch under the pinned transformers version.
    by_name["qwen3_moe_moe_patch"].apply(
        SimpleNamespace(model_type="qwen3_moe"), SimpleNamespace(true_on_policy_mode=False)
    )


@pytest.mark.parametrize("model_type", ["glm4_moe_lite", "nemotron_h", "qwen3", "qwen3_moe", "qwen3_vl"])
def test_model_type_verified_accepts_recorded_models(model_type, caplog):
    # [FUNCTION DOC] Parametrized over the five verified model types. The verification function,
    # given a config of that type and a rank-0 context, logs nothing (no records captured).
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.class_patches import check_model_type_verified

    check_model_type_verified(SimpleNamespace(model_type=model_type), SimpleNamespace(rank=0))

    assert not caplog.records


def test_verified_model_types_match_recorded_validation():
    # [FUNCTION DOC] Asserts the verified model types constant is exactly the frozen set of
    # glm4_moe_lite, nemotron_h, qwen3, qwen3_moe and qwen3_vl.
    from miles.backends.fsdp_utils.adaptations.class_patches import VERIFIED_MODEL_TYPES

    assert VERIFIED_MODEL_TYPES == frozenset({"glm4_moe_lite", "nemotron_h", "qwen3", "qwen3_moe", "qwen3_vl"})


def test_model_type_verified_warns_once_on_rank_zero(caplog):
    # [FUNCTION DOC] Applies the model-type-verified hook to an unverified type (qwen3_5_moe) on
    # rank 0. Asserts exactly one log record, whose text says the model type has no recorded FSDP
    # validation.
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS

    hook = next(h for h in _MODEL_PATCH_HOOKS if h.name == "model_type_verified")
    hook.apply(SimpleNamespace(model_type="qwen3_5_moe"), SimpleNamespace(rank=0))

    assert len(caplog.records) == 1
    assert "model_type='qwen3_5_moe' has no recorded FSDP validation" in caplog.text


def test_model_type_verified_is_silent_on_nonzero_rank(caplog):
    # [FUNCTION DOC] Applies the same hook to the unverified type on rank 1 and asserts nothing is
    # logged.
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS

    hook = next(h for h in _MODEL_PATCH_HOOKS if h.name == "model_type_verified")
    hook.apply(SimpleNamespace(model_type="qwen3_5_moe"), SimpleNamespace(rank=1))

    assert not caplog.records


def test_packed_seq_context_boundaries():
    # [FUNCTION DOC] Asserts the packed-sequence context function returns none for: no position
    # ids, a single document (positions 0..7 never reset), a one-dimensional tensor, and a batch
    # larger than one. For position ids [[0,1,2,0,1,0,1,2,3]] (documents of lengths 3, 2, 4) it
    # returns a context with cumulative sequence lengths [0,3,5,9] (32-bit integers), sequence
    # indices [[0,0,0,1,1,2,2,2,2]] (32-bit integers, shape (1, 9)) and maximum sequence length 4.
    # The shared boundary derivation (formerly duplicated verbatim in nemotron_h.py + qwen3_5_moe.py).
    from miles.backends.fsdp_utils.adaptations.packing.boundaries import packed_seq_context

    # single document / non-packed / wrong shape -> None (packing is a no-op)
    assert packed_seq_context(None) is None
    assert packed_seq_context(torch.arange(8).view(1, 8)) is None  # one doc, never resets to 0
    assert packed_seq_context(torch.arange(8)) is None  # not [1, T]
    assert packed_seq_context(torch.zeros(2, 4, dtype=torch.long)) is None  # batch > 1

    # three packed docs of length 3, 2, 4 -> position_ids reset to 0 at each start
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    ctx = packed_seq_context(pos)
    assert ctx is not None
    assert ctx.cu_seqlens.tolist() == [0, 3, 5, 9]
    assert ctx.cu_seqlens.dtype == torch.int32
    assert ctx.seq_idx.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2]]
    assert ctx.seq_idx.dtype == torch.int32
    assert ctx.seq_idx.shape == (1, 9)
    assert ctx.max_seqlen == 4


def test_nemotron_attention_reuses_precomputed_max_seqlen(monkeypatch):
    # [FUNCTION DOC] Installs a fake flash-attention module (via monkeypatch) whose variable-length
    # function records its keyword arguments and returns the query. The packed-sequence context
    # getter in the Nemotron-H module is patched to return a context with an unindexable
    # cumulative-length object (indexing raises an assertion error), no sequence indices and
    # maximum sequence length 3. Patches the attention and causal-LM forwards of dummy classes via
    # the module's patch functions, runs the dummy causal LM on a (1, 3, 2) ones tensor with zero
    # position ids, and asserts: the output has shape (1, 3, 2); flash attention received the very
    # same cumulative-length object as the query cumulative lengths, and 3 as both maximum query
    # and key sequence lengths (the maximum was reused, not recomputed).
    import sys
    from types import ModuleType, SimpleNamespace

    from miles.backends.fsdp_utils.models import nemotron_h

    flash_calls = {}
    flash_attn = ModuleType("flash_attn")

    def flash_attn_varlen_func(q, k, v, **kwargs):
        # [FUNCTION DOC] HELPER: Records the keyword arguments into the enclosing dictionary and
        # returns the query unchanged.
        flash_calls.update(kwargs)
        return q

    flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    class UnreadableCuSeqlens:
        # [CLASS DOC] HELPER: Marker object that fails the test if anything indexes into it.
        def __getitem__(self, key):
            # [FUNCTION DOC] HELPER: Always raises an assertion error saying attention must not
            # recompute the maximum sequence length from cumulative lengths.
            raise AssertionError("attention must not recompute max_seqlen from cu_seqlens")

    cu_seqlens = UnreadableCuSeqlens()
    ctx = SimpleNamespace(cu_seqlens=cu_seqlens, seq_idx=None, max_seqlen=3)
    monkeypatch.setattr(nemotron_h, "packed_seq_context", lambda position_ids: ctx)

    class DummyMixer(torch.nn.Module):
        # [CLASS DOC] HELPER: Empty placeholder mixer class passed to the causal-LM patcher.
        pass

    class DummyAttention(torch.nn.Module):
        # [CLASS DOC] HELPER: Fake attention layer with head dimension 2 and identity projections
        # whose original forward fails the test (the patched packed path must be used).
        def __init__(self):
            # [FUNCTION DOC] HELPER: Sets head dimension 2 and identity query/key/value/output
            # projections.
            super().__init__()
            self.head_dim = 2
            self.q_proj = torch.nn.Identity()
            self.k_proj = torch.nn.Identity()
            self.v_proj = torch.nn.Identity()
            self.o_proj = torch.nn.Identity()

        def forward(self, hidden_states, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises an assertion error (packed attention must use
            # the variable-length flash attention function instead).
            raise AssertionError("packed attention should use flash_attn_varlen_func")

    class DummyCausalLM(torch.nn.Module):
        # [CLASS DOC] HELPER: Fake causal LM holding one attention layer.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Creates the attention layer.
            super().__init__()
            self.attn = DummyAttention()

        def forward(self, hidden_states, position_ids=None):
            # [FUNCTION DOC] HELPER: Runs the attention layer on the hidden states; the position
            # ids are unused here (a patch supplies them to the packing context).
            return self.attn(hidden_states)

    nemotron_h._patch_attn_forward(DummyAttention)
    nemotron_h._patch_causallm_forward(DummyCausalLM, DummyMixer, DummyAttention)

    model = DummyCausalLM()
    output, _ = model(torch.ones(1, 3, 2), position_ids=torch.zeros(1, 3, dtype=torch.long))

    assert output.shape == (1, 3, 2)
    assert flash_calls["cu_seqlens_q"] is cu_seqlens
    assert flash_calls["max_seqlen_q"] == 3
    assert flash_calls["max_seqlen_k"] == 3


def test_nemotron_pattern_to_list_repair():
    # [FUNCTION DOC] Saves the config class's real pattern-to-list static method. First installs a
    # healthy version (mapping M, E, *, - to mamba, moe, attention, mlp) and asserts the repair
    # leaves it untouched (same object). Then installs a broken version that drops the "-"
    # character and asserts that after repair the pattern "M-*E" maps to mamba, mlp, attention,
    # moe. Always restores the original method in a finally block.
    # sglang's import monkeypatches NemotronHConfig._pattern_to_list to drop unmapped chars, deleting
    # every '-' (MLP) layer; the config-time repair must restore the full mapping, and must leave a
    # healthy implementation untouched.
    from types import SimpleNamespace

    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    from miles.backends.fsdp_utils.adaptations.specs.nemotron_h import _repair_pattern_to_list

    hf_config = SimpleNamespace(model_type="nemotron_h")
    original = NemotronHConfig.__dict__["_pattern_to_list"]
    healthy = staticmethod(
        lambda pattern: [{"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}[c] for c in pattern]
    )
    broken = staticmethod(
        lambda pattern: [{"M": "mamba", "E": "moe", "*": "attention"}[c] for c in pattern if c in "ME*"]
    )
    try:
        NemotronHConfig._pattern_to_list = healthy
        _repair_pattern_to_list(hf_config, None)
        assert NemotronHConfig.__dict__["_pattern_to_list"] is healthy

        NemotronHConfig._pattern_to_list = broken
        _repair_pattern_to_list(hf_config, None)
        assert NemotronHConfig._pattern_to_list("M-*E") == ["mamba", "mlp", "attention", "moe"]
    finally:
        NemotronHConfig._pattern_to_list = original


def test_packing_registry():
    # [FUNCTION DOC] Queries the packing patches by (config, lifetime) for four configs: a gated
    # delta net hybrid (qwen3_5_moe with linear and full attention layers) gives exactly
    # "gated_deltanet_packing" at config lifetime and nothing at post-load; Nemotron-H gives
    # exactly "nemotron_h_packing" at post-load and nothing at config; GLM4 MoE-lite, dense qwen3
    # and a missing config give nothing at either lifetime.
    # The unified packing registry dispatches per (model_type, lifetime); GDN is config-lifetime,
    # NemotronH is post-load-lifetime, and archs that pack natively / don't pack match nothing.
    from types import SimpleNamespace

    from miles.backends.fsdp_utils.adaptations.packing import get_packing_patches

    gdn = SimpleNamespace(model_type="qwen3_5_moe", layer_types=["linear_attention", "full_attention"])
    nemo = SimpleNamespace(model_type="nemotron_h")
    glm = SimpleNamespace(model_type="glm4_moe_lite", layer_types=["full_attention"])
    dense = SimpleNamespace(model_type="qwen3", layer_types=["full_attention"])

    def names(cfg, lifetime):
        # [FUNCTION DOC] HELPER: Returns the set of names of the packing patches that apply to the
        # config at the given lifetime.
        return {p.name for p in get_packing_patches(cfg, lifetime)}

    # GatedDeltaNet: config lifetime only
    assert names(gdn, "config") == {"gated_deltanet_packing"}
    assert names(gdn, "post_load") == set()
    # NemotronH: post-load lifetime only
    assert names(nemo, "post_load") == {"nemotron_h_packing"}
    assert names(nemo, "config") == set()
    # glm4_moe_lite (native MLA varlen) and dense qwen3: no packing patch at either lifetime
    for cfg in (glm, dense, None):
        assert names(cfg, "config") == set()
        assert names(cfg, "post_load") == set()


from types import SimpleNamespace

import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from miles.backends.fsdp_utils.actor import FSDPTrainRayActor


def _model_cls(**config_fields):
    # [FUNCTION DOC] HELPER: Creates an FSDP train actor without its constructor, sets its HF
    # config to a simple namespace of the given fields, and returns the model class the actor's
    # model-class getter chooses.
    actor = object.__new__(FSDPTrainRayActor)
    actor.hf_config = SimpleNamespace(**config_fields)
    return actor._get_model_cls()


def test_native_vlm_routes_to_image_text_to_text():
    # [FUNCTION DOC] A qwen3_vl config with a vision config and no auto map selects the
    # image-text-to-text auto model class.
    # Qwen3-VL: multimodal, no auto_map, resolves through the transformers registry.
    assert _model_cls(model_type="qwen3_vl", vision_config={"depth": 24}) is AutoModelForImageTextToText


def test_remote_code_multimodal_without_i2t_in_auto_map_routes_to_causal_lm():
    # [FUNCTION DOC] A Kimi-K2.5-style config with a vision config but an auto map that has only
    # config, base model and causal-LM entries selects the causal-LM auto model class.
    # Kimi-K2.5 ships a vision_config but its remote code only maps AutoModelForCausalLM,
    # so asking for AutoModelForImageTextToText raises "Unrecognized configuration".
    cls = _model_cls(
        model_type="kimi_k25",
        vision_config={"init_pos_emb_height": 64},
        auto_map={
            "AutoConfig": "configuration_kimi.KimiK25Config",
            "AutoModel": "modeling_kimi.KimiK25Model",
            "AutoModelForCausalLM": "modeling_kimi.KimiK25ForCausalLM",
        },
    )
    assert cls is AutoModelForCausalLM


def test_remote_code_multimodal_declaring_i2t_routes_to_image_text_to_text():
    # [FUNCTION DOC] A remote-code config with a vision config whose auto map declares an
    # image-text-to-text entry selects the image-text-to-text auto model class.
    cls = _model_cls(
        model_type="some_remote_vlm",
        vision_config={"depth": 8},
        auto_map={"AutoModelForImageTextToText": "modeling_x.XForConditionalGeneration"},
    )
    assert cls is AutoModelForImageTextToText


def test_text_only_remote_code_routes_to_causal_lm():
    # [FUNCTION DOC] A text-only remote-code config with a causal-LM auto map entry selects the
    # causal-LM auto model class.
    cls = _model_cls(model_type="some_remote_lm", auto_map={"AutoModelForCausalLM": "modeling_x.XForCausalLM"})
    assert cls is AutoModelForCausalLM


def test_native_causal_lm_resolves_concrete_class():
    # [FUNCTION DOC] A plain qwen3 config (no auto map) resolves to the concrete Qwen3 causal-LM
    # class from the transformers package.
    assert _model_cls(model_type="qwen3") is transformers.Qwen3ForCausalLM
