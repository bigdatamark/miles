# [MODULE DOC] Unit tests for the LoRA helper that decides whether a list of target module names
# (or a single bare string) reaches mixture-of-experts leaf layers. Each test asserts truthiness
# of the helper's result for a group of target names.
"""targets_expert_leaves gates the MoE-specific multi-LoRA handling (permute-fusion
off, expert validations); a false negative silently skips those."""

from miles.utils.lora.utils import targets_expert_leaves


def test_mlp_leaf_names_target_experts():
    # [FUNCTION DOC] Asserts the dense MLP projection names (gate, up, down) and the two fused
    # linear layer names each count as targeting experts.
    # These names match the dense MLP and the routed experts alike.
    assert targets_expert_leaves(["gate_proj", "up_proj", "down_proj"])
    assert targets_expert_leaves(["linear_fc1"])
    assert targets_expert_leaves(["linear_fc2"])


def test_expert_scoped_wildcards_target_experts():
    # [FUNCTION DOC] Asserts a wildcard pattern scoped to the experts of any layer counts as
    # targeting experts.
    assert targets_expert_leaves(["*.layers.*.mlp.experts.linear_fc1"])


def test_attention_only_targets_do_not():
    # [FUNCTION DOC] Asserts attention-only names (fused qkv and output projection, and separate
    # q, k, v, o projections) do not count.
    assert not targets_expert_leaves(["linear_qkv", "linear_proj"])
    assert not targets_expert_leaves(["q_proj", "k_proj", "v_proj", "o_proj"])


def test_bulk_aliases_target_experts():
    # [FUNCTION DOC] Asserts the bulk aliases all, all-linear, all_linear and uppercase ALL each
    # count as targeting experts (case-insensitive; the alias is not expanded here). The alias is
    # the assertion message on failure.
    # "all" is only resolved by the later target-module conversion, so the alias itself counts.
    for alias in ("all", "all-linear", "all_linear", "ALL"):
        assert targets_expert_leaves([alias]), alias


def test_bare_string_is_accepted():
    # [FUNCTION DOC] Asserts a plain string (not a list) is accepted: the gate projection name
    # counts, the fused qkv name does not.
    assert targets_expert_leaves("gate_proj")
    assert not targets_expert_leaves("linear_qkv")


def test_empty_targets_do_not():
    # [FUNCTION DOC] Asserts none and an empty list do not count as targeting experts.
    assert not targets_expert_leaves(None)
    assert not targets_expert_leaves([])
