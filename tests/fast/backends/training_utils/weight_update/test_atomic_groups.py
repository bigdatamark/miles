# [MODULE DOC] CPU unit tests for the registry of atomic update groups in the HF weight namespace
# (weights that must be sent to the rollout engine together because the engine fuses them). Checks:
# deepseekv4 registers three fused pairs (q-a plus kv projection; compressor kv plus gate; indexer
# compressor kv plus gate) with exact keys and name suffixes; suffix matching by ends-with resolves
# uniquely across them; model-specific groups take precedence over the generic low-rank-query
# fallback; the fallback (given a q lora rank) yields a single pair of q-a and kv-a-with-mqa
# suffixes matching both deepseek and kimi-vl name prefixes; models without q lora (qwen3) or
# inkling register none. Registered as a CPU test in the first CPU stage.
"""Unit tests for the HF-namespace atomic group registry."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


class TestHfAtomicUpdateGroups:
    # [CLASS DOC] Tests for the group lookup function.
    def test_deepseekv4_registers_the_three_fused_pairs(self):
        # [FUNCTION DOC] Test: the mapping from group key to suffix tuple for deepseekv4 equals the
        # three expected entries exactly (keys wqkv_a, compressor_wkv_gate,
        # indexer_compressor_wkv_gate).
        groups = {g.key: g.suffixes for g in get_hf_atomic_update_groups("deepseekv4")}
        assert groups == {
            "wqkv_a": (".self_attn.wq_a.weight", ".self_attn.wkv.weight"),
            "compressor_wkv_gate": (".self_attn.compressor.wkv.weight", ".self_attn.compressor.wgate.weight"),
            "indexer_compressor_wkv_gate": (
                ".self_attn.indexer.compressor.wkv.weight",
                ".self_attn.indexer.compressor.wgate.weight",
            ),
        }

    def test_deepseekv4_suffixes_do_not_shadow_each_other(self):
        # [FUNCTION DOC] Test: for a name built from each registered suffix ("model.layers.0" plus
        # the suffix), exactly one registered suffix (itself) is matched by an ends-with check.
        """endswith matching must resolve uniquely across the three v4 pairs."""
        suffixes = [s for g in get_hf_atomic_update_groups("deepseekv4") for s in g.suffixes]
        sample = {s: f"model.layers.0{s}" for s in suffixes}
        for suffix, name in sample.items():
            matches = [s for s in suffixes if name.endswith(s)]
            assert matches == [suffix], f"{name} matched {matches}"

    def test_model_groups_take_precedence_over_q_lora(self):
        # [FUNCTION DOC] Test: for deepseekv4 with q lora rank 1024 the returned group keys are only
        # the three model-specific ones, in order (the generic fallback is not appended).
        groups = get_hf_atomic_update_groups("deepseekv4", q_lora_rank=1024)
        assert [g.key for g in groups] == ["wqkv_a", "compressor_wkv_gate", "indexer_compressor_wkv_gate"]

    def test_q_lora_fallback(self):
        # [FUNCTION DOC] Test: for deepseekv3 with q lora rank 1024 there is one group whose suffixes
        # are the q-a projection and the kv-a-with-mqa projection weights; the first suffix matches
        # names under both the plain "model.layers.N" prefix and the "language_model.model.layers.N"
        # prefix.
        groups = get_hf_atomic_update_groups("deepseekv3", q_lora_rank=1024)
        assert [g.suffixes for g in groups] == [
            ((".self_attn.q_a_proj.weight", ".self_attn.kv_a_proj_with_mqa.weight"))
        ]
        # Suffix matching covers both deepseek and kimi-vl prefixes.
        for prefix in ("model.layers.3", "language_model.model.layers.3"):
            assert f"{prefix}.self_attn.q_a_proj.weight".endswith(groups[0].suffixes[0])

    def test_no_q_lora_no_groups(self):
        # [FUNCTION DOC] Test: qwen3 without a q lora rank returns an empty list.
        assert get_hf_atomic_update_groups("qwen3") == []

    def test_inkling_registers_none(self):
        # [FUNCTION DOC] Test: inkling returns an empty list.
        assert get_hf_atomic_update_groups("inkling") == []
