# [MODULE DOC]
# Megatron-to-Hugging Face converter for "Inkling" models. Reads the checkpoint's config.json
# (text config) for attention geometry (local sliding-window layers have their own head counts),
# the relative-position width, expert counts and intermediate size. Handles: de-sharding the fused
# attention q|k|v|r projection (tensor-parallel-gathered blocks interleaved per rank) into four
# separate weights; re-interleaving gate/up halves into alternating rows for the fused "w13"
# layout; gathering the router gate and shared gate into one tensor; routed experts split into
# gate/up/down; shared experts stacked across experts. Multi-piece groups are accumulated across
# calls in a module-level accumulator (the piece order may be arbitrary).
import json
import re
from functools import cache

import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix


@cache
def _load_text_cfg(hf_checkpoint: str) -> dict:
    # [FUNCTION DOC] HELPER: Memoized per checkpoint path. Reads config.json in the directory and
    # returns its "text_config" sub-mapping if present, else the whole config. Disk read.
    with open(f"{hf_checkpoint}/config.json") as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def _text_cfg(args) -> dict:
    # [FUNCTION DOC] HELPER: The cached text config for args.hf_checkpoint.
    return _load_text_cfg(args.hf_checkpoint)


def _attn_geometry(args, layer_idx: int) -> tuple[int, int, int, int]:
    # [FUNCTION DOC] HELPER: Returns (query heads, key/value heads, head dim, relative width) for
    # the layer. Layers whose index is in the config's local layer id list use the sliding-window
    # variants of the head counts and head dim; others the global ones. The relative width is shared.
    """(nh, nkv, hd, d_rel) for the layer; SWA layers use the swa_* head counts."""
    t = _text_cfg(args)
    is_local = layer_idx in set(t["local_layer_ids"])
    nh = t["swa_num_attention_heads"] if is_local else t["num_attention_heads"]
    nkv = t["swa_num_key_value_heads"] if is_local else t["num_key_value_heads"]
    hd = t["swa_head_dim"] if is_local else t["head_dim"]
    return nh, nkv, hd, t["d_rel"]


class _PieceAccumulator:
    # [CLASS DOC] HELPER: Buffer collecting pieces of a multi-tensor group keyed by (kind, layer),
    # each piece with a sub-index, until the expected count has arrived. Mutable state.
    """Collects multi-param groups (shared experts, gate halves) across convert calls."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with no buckets.
        self._buckets: dict[tuple[str, int], dict[int, torch.Tensor]] = {}

    def put(self, kind: str, layer: int, sub_idx: int, tensor: torch.Tensor, expected: int):
        # [FUNCTION DOC] HELPER: Stores the tensor under its sub-index (a repeated index overwrites).
        # Returns none while fewer than the expected number of distinct pieces are present. Once the
        # count reaches the expected number, removes the bucket and returns the pieces as a list
        # ordered by sub-index. Asserts the count is exactly the expected one (it can exceed it only
        # if a bucket was left over from a different expected value).
        """Stash one piece; return the ordered tensors once all `expected` arrived, else None."""
        bucket = self._buckets.setdefault((kind, layer), {})
        bucket[sub_idx] = tensor
        if len(bucket) < expected:
            return None
        assert len(bucket) == expected, f"{kind} layer {layer}: got {len(bucket)} pieces, expected {expected}"
        del self._buckets[(kind, layer)]
        return [bucket[i] for i in sorted(bucket)]


_ACC = _PieceAccumulator()


def _qkv_tp_size(args):
    # [FUNCTION DOC] HELPER: Tensor-parallel size from the global parallel state; on any exception
    # (state not set up) falls back to args.tensor_model_parallel_size, treating none/zero as 1.
    try:
        from miles.backends.training_utils.parallel import get_parallel_state

        return get_parallel_state().tp.size
    except Exception:
        return getattr(args, "tensor_model_parallel_size", 1) or 1


def _qkv_blocks_from_gathered(args, layer_idx: int, param: torch.Tensor):
    # [FUNCTION DOC] HELPER: Input is the fused projection gathered over tensor-parallel ranks with
    # layout, per rank, of query, key, value and relative blocks concatenated along rows. Expected
    # full row counts: query = heads*head dim, key = value = kv heads*head dim, relative = heads *
    # relative width. Asserts the row count equals their sum. With TP size 1 simply splits along
    # axis 0. Otherwise asserts divisibility, computes per-rank block sizes (kv heads per rank is
    # at least 1), asserts per-rank sizes times TP equal the total (not supported when kv heads are
    # replicated, i.e. fewer than TP), splits into per-rank shards, splits each shard into the four
    # blocks and concatenates each block kind across ranks in rank order. Returns a tuple (q, k, v,
    # r) of tensors.
    """Reconstruct full q|k|v|r from TP-all-gathered [q0|k0|v0|r0 | q1|k1|v1|r1 | ...] blocks."""
    nh, nkv, hd, d_rel = _attn_geometry(args, layer_idx)
    full = [nh * hd, nkv * hd, nkv * hd, nh * d_rel]
    full_rows = sum(full)
    rows = param.shape[0]

    assert rows == full_rows, (
        f"qkvr layer {layer_idx}: gathered rows {rows} != full q|k|v|r rows {full_rows} "
        f"(nh={nh} nkv={nkv} hd={hd} d_rel={d_rel}). all_gather_param should return the full "
        f"(un-sharded) weight; got a partial/oversized tensor."
    )

    tp = _qkv_tp_size(args)
    if tp == 1:
        return tuple(param.split(full, dim=0))

    assert full_rows % tp == 0, f"qkvr layer {layer_idx}: full rows {full_rows} not divisible by tp {tp}"
    nh_l = nh // tp
    nkv_l = max(1, nkv // tp)
    per = [nh_l * hd, nkv_l * hd, nkv_l * hd, nh_l * d_rel]
    assert sum(per) * tp == full_rows, (
        f"qkvr layer {layer_idx}: per-rank block {per} x tp {tp} = {sum(per)*tp} != full rows "
        f"{full_rows}. Likely nkv({nkv}) < tp({tp}) (kv-head replication), which this de-shard "
        f"does not handle -- the model keeps nkv>=tp at train time (global nkv=8, local nkv=16)."
    )
    shards = param.split(sum(per), dim=0)
    blk = [s.split(per, dim=0) for s in shards]
    return tuple(torch.cat([blk[i][j] for i in range(tp)], dim=0) for j in range(4))


def _reinterleave_w13(w13: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts a contiguous [gate rows; up rows] matrix (even number of rows,
    # asserted) into the interleaved layout where row 2i is gate row i and row 2i+1 is up row i;
    # implemented by stacking along a new axis 1 and reshaping to the original shape.
    two_i = w13.shape[0]
    assert two_i % 2 == 0, f"w13 first dim must be even, got {two_i}"
    half = two_i // 2
    gate, up = w13[:half], w13[half:]
    return torch.stack((gate, up), dim=1).reshape(w13.shape)


_GLOBAL_MAPPING = {
    "embedding.word_embeddings.weight": "model.llm.embed.weight",
    "embedding.embed_norm.weight": "model.llm.embed_norm.weight",
    "decoder.final_layernorm.weight": "model.llm.norm.weight",
    "output_layer.weight": "model.llm.unembed.weight",
}


def convert_inkling_to_hf(args, name, param):
    # [FUNCTION DOC]
    # Input: args (HF checkpoint path for config), Megatron name (any number of leading "module."
    # prefixes are stripped), tensor. Output: list of (HF name, tensor); an empty list means the
    # piece was buffered awaiting siblings. Global names map to model.llm.embed / embed_norm / norm /
    # unembed. Names that are not decoder layer names are passed through unchanged (with the prefix
    # stripped). Layer names (HF prefix model.llm.layers.<idx>.): fused QKV weight is de-sharded into
    # wq_du, wk_dv, wv_dv, wr_du weights; QKV-fused layer norm and pre-MLP layernorm and the MLP
    # fused layer norm map to attn_norm / mlp_norm; output projection to wo_ud; query/key norms,
    # relative projection, key/value/attention/MLP short-convolution weights map directly; dense
    # MLP first linear is re-interleaved to w13_dn, second linear to w2_md, global scale direct;
    # router weight (piece 0) and shared gate (piece 1) are buffered and emitted as one gate weight
    # (concatenated along axis 0) once both arrive; router expert bias to gate bias; router global
    # scale to gate global scale; routed expert first linear: asserts even rows and, if the config
    # gives an intermediate size, rows equal twice that (ETP must be 1), splits in half into gate
    # and up projections for expert index j; routed expert second linear to down projection;
    # shared experts: first linear (re-interleaved) and second linear pieces are buffered per layer
    # until the configured shared-expert count have arrived, then emitted stacked along a new axis 0
    # as shared_w13_weight and shared_w2_weight. Unhandled layer names raise a value error.
    # KNOWN ISSUE: the buffer is module-level global state shared across calls and not reset if a
    # group never completes.
    mc = strip_param_name_prefix(name)

    if mc in _GLOBAL_MAPPING:
        return [(_GLOBAL_MAPPING[mc], param)]

    m = re.match(r"decoder\.layers\.(\d+)\.(.+)", mc)
    if m is None:
        return [(mc, param)]
    layer = int(m.group(1))
    rest = m.group(2)
    hp = f"model.llm.layers.{layer}."

    if rest == "self_attention.linear_qkv.weight":
        q, k, v, r = _qkv_blocks_from_gathered(args, layer, param)
        return [
            (hp + "attn.wq_du.weight", q),
            (hp + "attn.wk_dv.weight", k),
            (hp + "attn.wv_dv.weight", v),
            (hp + "attn.wr_du.weight", r),
        ]
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        return [(hp + "attn_norm.weight", param)]
    if rest == "self_attention.linear_proj.weight":
        return [(hp + "attn.wo_ud.weight", param)]
    if rest == "self_attention.q_norm.weight":
        return [(hp + "attn.q_norm.weight", param)]
    if rest == "self_attention.k_norm.weight":
        return [(hp + "attn.k_norm.weight", param)]
    if rest == "self_attention.rel_proj":
        return [(hp + "attn.rel_logits_proj.proj", param)]
    if rest == "self_attention.k_sconv.weight":
        return [(hp + "attn.k_sconv.weight", param)]
    if rest == "self_attention.v_sconv.weight":
        return [(hp + "attn.v_sconv.weight", param)]
    if rest == "self_attention.attn_sconv.weight":
        return [(hp + "attn_sconv.weight", param)]

    if rest == "pre_mlp_layernorm.weight":
        return [(hp + "mlp_norm.weight", param)]
    if rest == "mlp.mlp_sconv.weight":
        return [(hp + "mlp_sconv.weight", param)]

    if rest == "mlp.linear_fc1.layer_norm_weight":
        return [(hp + "mlp_norm.weight", param)]
    if rest == "mlp.linear_fc1.weight":
        return [(hp + "mlp.w13_dn.weight", _reinterleave_w13(param))]
    if rest == "mlp.linear_fc2.weight":
        return [(hp + "mlp.w2_md.weight", param)]
    if rest == "mlp.global_scale":
        return [(hp + "mlp.global_scale", param)]

    gate_sub = {"mlp.router.weight": 0, "mlp.router.shared_gate": 1}.get(rest)
    if gate_sub is not None:
        out = _ACC.put("gate", layer, gate_sub, param, 2)
        if out is None:
            return []
        return [(hp + "mlp.gate.weight", torch.cat(out, dim=0))]
    if rest == "mlp.router.expert_bias":
        return [(hp + "mlp.gate.bias", param)]
    if rest == "mlp.router.global_scale":
        return [(hp + "mlp.gate.global_scale", param)]

    em = re.match(r"mlp\.experts\.(linear_fc1|linear_fc2)\.weight(\d+)", rest)
    if em is not None:
        which, j = em.group(1), int(em.group(2))
        if which == "linear_fc1":
            two_i = param.shape[0]
            assert two_i % 2 == 0, (
                f"routed expert {j} layer {layer}: linear_fc1 rows {two_i} not even; "
                f"expected contiguous [gate; up] (2*intermediate)"
            )
            inter = _text_cfg(args).get("intermediate_size")
            assert inter is None or two_i == 2 * inter, (
                f"routed expert {j} layer {layer}: linear_fc1 rows {two_i} != 2*intermediate_size "
                f"({2 * inter}). The gathered expert is not the full contiguous [gate; up] this "
                f"converter assumes (ETP must be 1; no TP split on the expert intermediate dim)."
            )
            gate, up = param.chunk(2, dim=0)
            return [
                (hp + f"mlp.experts.{j}.gate_proj.weight", gate),
                (hp + f"mlp.experts.{j}.up_proj.weight", up),
            ]
        else:
            return [(hp + f"mlp.experts.{j}.down_proj.weight", param)]

    sm = re.match(r"mlp\.shared_experts\.experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight", rest)
    if sm is not None:
        e, which = int(sm.group(1)), sm.group(2)
        ns = _text_cfg(args)["n_shared_experts"]
        if which == "linear_fc1":
            out = _ACC.put("shared_fc1", layer, e, _reinterleave_w13(param), ns)
            if out is None:
                return []
            return [(hp + "mlp.shared_experts.shared_w13_weight", torch.stack(out, dim=0))]
        else:
            out = _ACC.put("shared_fc2", layer, e, param, ns)
            if out is None:
                return []
            return [(hp + "mlp.shared_experts.shared_w2_weight", torch.stack(out, dim=0))]

    raise ValueError(f"convert_inkling_to_hf: unhandled param name {name!r} (rest={rest!r})")
