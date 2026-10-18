# [MODULE DOC]
# Tests of lora target resolution and native adapter export for the inkling model plugin: text and
# multimodal configs select the same hf mlp targets, the export plans for dense mlp and experts
# produce hf named lora weights recognised by the adapter target helper, incomplete layouts are
# rejected, and a legacy config resolves to the same targets as the native one.
from types import SimpleNamespace

import pytest
import torch

from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules
from miles_plugins.models.inkling.lora import _export_dense_mlp, _export_experts, resolve_inkling_adapter_targets


class _LocalGather:
    # [CLASS DOC] HELPER: Stand-in for a distributed gather that gathers nothing: adding a tensor
    # returns a handle whose get returns that same tensor.
    def add(self, group, tensor, dim):
        # [FUNCTION DOC] HELPER: Ignores the group and dim and returns an object with a get method that
        # returns the given tensor unchanged.
        return SimpleNamespace(get=lambda: tensor)


@pytest.mark.parametrize("multimodal", [False, True], ids=["text", "multimodal"])
def test_hf_mlp_selection_matches_existing_native_export(multimodal):
    # [FUNCTION DOC] Parametrized over a text config and a multimodal config wrapping it. The text
    # config has a dense then sparse layer pattern and no shared experts. Steps: resolve hf targets;
    # assert the adapter target resolver returns the all-linear selection; assert dense gate projection
    # and fused expert gate-up targets exist under the model or language-model prefix. Then builds fake
    # dense and expert modules with 2 by 2 ones tensors and runs both export functions with a local
    # gather; evaluates callable values; asserts the fused gate-up module name appears in the adapter
    # target modules, its lora A weight is the identical tensor and its lora B weight equals it. Finally
    # removes the up-projection targets and asserts the resolver raises an assertion error mentioning a
    # complete adapter layout.
    config = dict(model_type="inkling_text", mlp_layer_types=["dense", "sparse"], n_shared_experts=0)
    if multimodal:
        config = dict(model_type="inkling_mm_model", text_config=config)
    hf_targets = resolve_hf_lora_targets(config)
    assert resolve_inkling_adapter_targets(config, hf_targets) == "all-linear"
    prefix = "model.language_model" if multimodal else "model"
    assert f"{prefix}.layers.*.mlp.gate_proj" in hf_targets
    assert f"{prefix}.layers.*.mlp.experts.gate_up_proj" in hf_targets
    tensor = torch.ones(2, 2)
    dense = SimpleNamespace(
        hf_prefix="language_model.layers.0.mlp.",
        load_meta={"i_loc": 1},
        fc1_A=tensor,
        fc1_B=tensor,
        fc2_A=tensor,
        fc2_B=tensor,
    )
    experts = SimpleNamespace(
        hf_prefix="language_model.layers.1.mlp.experts.",
        **{f"w{projection}_{factor}": tensor for projection in (1, 2, 3) for factor in ("A", "B")},
    )
    plan = _export_dense_mlp(dense, _LocalGather()) + _export_experts(experts, _LocalGather())
    weights = {name: value() if callable(value) else value for name, value in plan}
    assert "language_model.layers.0.mlp.gate_up_proj" in get_adapter_target_modules(weights)
    assert weights["language_model.layers.0.mlp.gate_up_proj.lora_A.weight"] is tensor
    assert torch.equal(weights["language_model.layers.0.mlp.gate_up_proj.lora_B.weight"], tensor)
    with pytest.raises(AssertionError, match="complete adapter layout"):
        resolve_inkling_adapter_targets(config, [target for target in hf_targets if not target.endswith(".up_proj")])


def test_legacy_config_selects_the_same_native_adapters():
    # [FUNCTION DOC] Builds a legacy config (dense layer index 1, two layers, one shared expert) and
    # the equivalent native config; asserts both resolve to identical hf targets, and that the adapter
    # target resolver accepts the legacy targets without raising.
    legacy = dict(model_type="inkling_model", dense_mlp_idx=1, num_hidden_layers=2, n_shared_experts=1)
    native = dict(model_type="inkling_text", mlp_layer_types=["dense", "sparse"], n_shared_experts=1)
    assert resolve_hf_lora_targets(legacy) == resolve_hf_lora_targets(native)
    resolve_inkling_adapter_targets(legacy, resolve_hf_lora_targets(legacy))
