# [MODULE DOC] CPU unit tests of the resolution of Hugging Face LoRA target module names to
# Megatron module patterns (and, in canonical mode, to adapter target names). Uses small fake
# parameter mappings (megatron parameter pattern to one or several HF parameter names, with a
# star wildcard standing for layer and expert indices) and checks: scoped attention selection
# excludes multi-token-prediction layers; selecting part of a fused projection is rejected unless
# canonical mode splits it; grouped and sequential expert layouts are alternatives; unknown HF
# targets fail; one-to-one mappings keep the bridge's module name; an absent fused alternative does
# not reject a selection; and canonical selection cannot mix different projections across layers.
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.lora.target_modules import resolve_megatron_lora_targets
from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets


class _Mapping(SimpleNamespace):
    # [CLASS DOC] HELPER: Fake weight mapping holding a Megatron parameter pattern and either one HF
    # parameter pattern string or a dictionary of several (a fused parameter); star wildcards in the
    # patterns are filled in order from captured values.
    def resolve(self, captures):
        # [FUNCTION DOC] HELPER: Substitutes the captured values, in order, for successive star
        # wildcards in the Megatron pattern and in each HF pattern, and returns a new fake mapping
        # built from the expanded names (a single HF name stays a string, several become a dictionary).
        def expand(name):
            # [FUNCTION DOC] HELPER: Replaces the first star in the name with each captured value
            # in turn (one star per capture) and returns the result.
            for value in captures:
                name = name.replace("*", value, 1)
            return name

        hf = [self.hf_param] if isinstance(self.hf_param, str) else self.hf_param.values()
        return _mapping(expand(self.megatron_param), *(expand(name) for name in hf))


def _mapping(megatron, *hf):
    # [FUNCTION DOC] HELPER: Builds a fake mapping from a Megatron pattern and one or more HF
    # patterns: one HF name is kept as a plain string, several are stored as a dictionary keyed by
    # position.
    return _Mapping(megatron_param=megatron, hf_param=hf[0] if len(hf) == 1 else dict(enumerate(hf)))


def _resolve(targets, mappings, parameter_names, *, canonical=False, hf_mapping=None):
    # [FUNCTION DOC] HELPER: Thin wrapper calling the target-resolution function with the given
    # HF targets, fake mappings, the parameter names as a set, an empty HF weight mapping by default,
    # and the canonical flag. Returns its result (a collection of Megatron module patterns, or adapter
    # names in canonical mode); assertion errors from the function propagate.
    return resolve_megatron_lora_targets(
        targets,
        mappings,
        parameter_names=set(parameter_names),
        hf_mapping=hf_mapping or HfWeightMapping(frozenset()),
        canonical=canonical,
    )


_QKV = _mapping(
    "decoder.layers.*.self_attention.linear_qkv.weight",
    *(f"model.layers.*.self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
)


@pytest.mark.parametrize(
    "mtp_source",
    ["mtp.layers.0.self_attn.o_proj.weight", "model.layers.1.self_attn.o_proj.weight"],
    ids=["separate-namespace", "appended-layer"],
)
def test_scoped_attention_excludes_mtp(mtp_source):
    # [FUNCTION DOC] Parametrized over two layouts for the multi-token-prediction output projection's
    # HF name (separate namespace, or appended as an extra layer of the main model). Targets the
    # "attn" scope of a qwen3 model, with fused qkv, output projection and mtp output projection
    # mappings and three Megatron parameter names; the HF weight mapping lists the main layer 0 q, k, v,
    # o names. Asserts the selection is exactly the main layer's qkv and output-projection module
    # patterns, i.e. the mtp projection is excluded.
    targets = resolve_hf_lora_targets({"model_type": "qwen3"}, target_modules=["attn"])
    output = _mapping("decoder.layers.*.self_attention.linear_proj.weight", "model.layers.*.self_attn.o_proj.weight")
    mtp = _mapping("mtp.layers.*.self_attention.linear_proj.weight", mtp_source)
    megatron_parameters = [
        "decoder.layers.0.self_attention.linear_qkv.weight",
        "decoder.layers.0.self_attention.linear_proj.weight",
        "mtp.layers.0.self_attention.linear_proj.weight",
    ]
    hf_mapping = HfWeightMapping(frozenset(f"model.layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o")))
    selected = _resolve(targets, [_QKV, output, mtp], megatron_parameters, hf_mapping=hf_mapping)
    assert set(selected) == {
        "decoder.layers.*.self_attention.linear_qkv",
        "decoder.layers.*.self_attention.linear_proj",
    }


def test_fused_selection_cannot_silently_expand():
    # [FUNCTION DOC] Selecting only the q projection of a fused qkv module raises an assertion error
    # saying all HF targets are required; in canonical mode the same selection yields only the
    # separate q adapter module pattern.
    targets = ["model.layers.*.self_attn.q_proj"]
    with pytest.raises(AssertionError, match="requires all HF targets"):
        _resolve(targets, [_QKV], ["decoder.layers.0.self_attention.linear_qkv.weight"])
    adapter_targets = _resolve(targets, [_QKV], ["decoder.layers.0.self_attention.linear_qkv.weight"], canonical=True)
    assert adapter_targets == ["decoder.layers.*.self_attention.linear_q"]


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "sequential"])
def test_expert_representations_are_alternatives(grouped):
    # [FUNCTION DOC] Parametrized over grouped and sequential expert layouts. Two mappings describe
    # the grouped and sequential Megatron names for the same HF expert down projection. With a parameter
    # name present in only one layout, asserts exactly one pattern is selected and it contains the
    # sequential local-experts marker iff the layout is sequential; then with a parameter name that
    # matches neither, asserts an assertion error saying no Megatron modules matched.
    target = "model.layers.*.mlp.experts.*.down_proj"
    mappings = [
        _mapping("decoder.layers.*.mlp.experts.linear_fc2.weight*", target + ".weight"),
        _mapping("decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight", target + ".weight"),
    ]
    module = "decoder.layers.0.mlp.experts." + ("linear_fc2" if grouped else "local_experts.0.linear_fc2")
    parameter = module + (".weight0" if grouped else ".weight")
    selected = _resolve([target], mappings, [parameter])
    assert len(selected) == 1
    assert ("local_experts" in next(iter(selected))) != grouped
    with pytest.raises(AssertionError, match="no Megatron modules"):
        _resolve([target], mappings, ["decoder.layers.0.mlp.linear_fc2.weight"])


def test_missing_hf_mapping_is_not_a_megatron_passthrough():
    # [FUNCTION DOC] An HF target with no mapping at all raises an assertion error saying no Megatron
    # modules matched (the name is not used as a Megatron module name directly).
    with pytest.raises(AssertionError, match="no Megatron modules"):
        _resolve(
            ["model.layers.*.self_attn.unknown_proj"], [_QKV], ["decoder.layers.0.self_attention.linear_qkv.weight"]
        )


def test_one_to_one_mapping_keeps_bridge_module_name():
    # [FUNCTION DOC] For a one-to-one mapping of an HF output projection to a bridge-named Megatron
    # output projection, canonical resolution yields exactly that module pattern (no renaming).
    target = "model.layers.*.self_attn.o_proj"
    mapping = _mapping("decoder.layers.*.self_attention.output_projection.weight", target + ".weight")
    adapter_targets = _resolve(
        [target], [mapping], ["decoder.layers.0.self_attention.output_projection.weight"], canonical=True
    )
    assert adapter_targets == ["decoder.layers.*.self_attention.output_projection"]


def test_absent_fused_alternative_does_not_reject_selection():
    # [FUNCTION DOC] Two alternative mappings cover the q projection: a fused qkv one and a separate
    # q one. With only the separate parameter present in the model and the target "q_proj", asserts
    # the fused alternative being absent does not cause rejection and the selection is the separate q
    # module pattern.
    mappings = [
        _mapping(
            "decoder.layers.*.self_attention.fused_qkv.weight",
            *(f"model.layers.*.self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
        ),
        _mapping("decoder.layers.*.self_attention.linear_q.weight", "model.layers.*.self_attn.q_proj.weight"),
    ]
    selected = _resolve(["q_proj"], mappings, ["decoder.layers.0.self_attention.linear_q.weight"])
    assert set(selected) == {"decoder.layers.*.self_attention.linear_q"}


def test_canonical_selection_cannot_expand_across_layers():
    # [FUNCTION DOC] Selecting q in layer 0 and k in layer 1 of a fused qkv module in canonical mode
    # raises an assertion error saying the targets are different projections.
    with pytest.raises(AssertionError, match="different projections"):
        _resolve(
            ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.k_proj"],
            [_QKV],
            [f"decoder.layers.{layer}.self_attention.linear_qkv.weight" for layer in range(2)],
            canonical=True,
        )
