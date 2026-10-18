# [MODULE DOC]
# Unit tests for the pure-Python LoRA target selection helpers: parsing a comma separated selector
# string, resolving selectors (the attn, mlp, unembed and all-linear keywords or explicit module
# names) against a model config dictionary into scoped glob patterns, excluding targets, and
# expanding paired gate/up projections to packed expert projections. No model is built; configs are
# minimal dictionaries holding only a model type.
"""LoRA selection uses model defaults and preserves explicit user choices."""

import pytest

from miles.utils.lora.hf_lora_targets import (
    exclude_hf_lora_targets,
    expand_packed_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)


@pytest.mark.parametrize("selection", [None, ["all-linear"], ["attn", "mlp"], ["attn", "mlp", "attn"]])
def test_missing_targets_and_all_linear_use_model_defaults(selection):
    # [FUNCTION DOC] Parametrized over no selection, the all-linear keyword, the attn plus mlp keywords
    # and the same with a duplicated keyword. For a qwen3 config, asserts every selection resolves to
    # the same list as the all-linear selection, that this list contains the scoped self-attention
    # query projection and the scoped mlp gate projection, and that the output head is not included.
    config = {"model_type": "qwen3"}
    targets = resolve_hf_lora_targets(config, target_modules=selection)
    assert targets == resolve_hf_lora_targets(config, target_modules=["all-linear"])
    assert "model.layers.*.self_attn.q_proj" in targets
    assert "model.layers.*.mlp.gate_proj" in targets
    assert "lm_head" not in targets


@pytest.mark.parametrize(
    "config,value,expected",
    [
        ({"model_type": "custom"}, "q_proj, k_proj", ["q_proj", "k_proj"]),
        ({"model_type": "custom"}, ["q_proj", "k_proj"], ["q_proj", "k_proj"]),
        ({"model_type": "qwen3"}, "unembed,lm_head", ["lm_head"]),
        (
            {"model_type": "qwen3"},
            "attn, lm_head",
            [f"model.layers.*.self_attn.{name}_proj" for name in ("q", "k", "v", "o")] + ["lm_head"],
        ),
        (
            {"model_type": "gpt_oss"},
            "mlp",
            ["model.layers.*.mlp.experts.gate_up_proj", "model.layers.*.mlp.experts.down_proj"],
        ),
        ({"model_type": "custom"}, "model.mlp", ["model.mlp"]),
    ],
)
def test_explicit_targets_preserve_user_choices(config, value, expected):
    # [FUNCTION DOC] Parametrized over (config, selector, expected list) cases: for an unknown model
    # type, comma/space separated or list selectors are kept verbatim in order; for qwen3, the unembed
    # keyword resolves to the output head, and the attn keyword expands to the four scoped attention
    # projections (q, k, v, o) followed by any explicit extra module; for gpt-oss the mlp keyword
    # resolves to the two packed expert projections; a dotted custom module name is kept as is.
    # Asserts the resolved target list equals the expected list exactly.
    assert resolve_hf_lora_targets(config, target_modules=parse_lora_targets(value)) == expected


def test_exclude_leaf_applies_to_scoped_model_defaults():
    # [FUNCTION DOC] Resolves the default qwen3 targets, excludes the bare leaf names output and down
    # projection, and asserts the scoped attention output and mlp down patterns are removed while the
    # scoped query projection is kept (a bare leaf name excludes the matching scoped pattern).
    targets = resolve_hf_lora_targets({"model_type": "qwen3"})
    selected = exclude_hf_lora_targets(targets, parse_lora_targets("o_proj, down_proj"))
    assert "model.layers.*.self_attn.o_proj" not in selected
    assert "model.layers.*.mlp.down_proj" not in selected
    assert "model.layers.*.self_attn.q_proj" in selected


def test_exclude_all_rejects_empty_selection():
    # [FUNCTION DOC] Excluding every selected target must raise an assertion error whose message says
    # the selection is empty after exclusion.
    with pytest.raises(AssertionError, match="empty after"):
        exclude_hf_lora_targets(["q_proj", "k_proj"], ["q_proj", "k_proj"])


def test_nonexistent_exclusion_does_not_change_selection():
    # [FUNCTION DOC] Excluding a name that matches nothing returns the original selection unchanged
    # and in the same order.
    assert exclude_hf_lora_targets(["q_proj", "k_proj"], ["nonexistent"]) == ["q_proj", "k_proj"]


def test_empty_selector_is_rejected():
    # [FUNCTION DOC] Parsing a selector string with an empty entry between two commas must raise an
    # assertion error whose message mentions empty entries.
    with pytest.raises(AssertionError, match="empty entries"):
        parse_lora_targets("q_proj,,k_proj")


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("scoped", [True, False])
def test_paired_targets_include_packed_experts(dense, scoped, caplog):
    # [FUNCTION DOC] Parametrized over whether dense mlp projections also exist in the module list and
    # whether the targets carry the scoped expert prefix. The module list always holds the packed
    # expert gate-up and down projections of layer 0 (plus dense gate, up, down when dense is set).
    # The targets are gate, up and down projections (prefixed when scoped). Asserts that expanding
    # the targets yields the original three targets when dense and unscoped, otherwise only the
    # (prefixed) down projection, always followed by the (prefixed) packed gate-up projection, and
    # that an info log says paired gate/up projections were expanded. Selecting only the gate
    # projection is returned unchanged, and when the module list has only a dense gate projection
    # the targets are returned unchanged.
    modules = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"]
    if dense:
        modules += [f"model.layers.0.mlp.{name}_proj" for name in ("gate", "up", "down")]
    prefix = "model.layers.*.mlp.experts." if scoped else ""
    targets = [prefix + name + "_proj" for name in ("gate", "up", "down")]
    expected = targets if dense and not scoped else [prefix + "down_proj"]
    assert expand_packed_hf_lora_targets(targets, modules) == expected + [prefix + "gate_up_proj"]
    assert "Expanding paired gate_proj/up_proj" in caplog.text
    assert expand_packed_hf_lora_targets(targets[:1], modules) == targets[:1]
    assert expand_packed_hf_lora_targets(targets, ["model.layers.0.mlp.gate_proj"]) == targets
