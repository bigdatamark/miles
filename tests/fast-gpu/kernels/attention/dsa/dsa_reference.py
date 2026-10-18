# [MODULE DOC] HELPER: Test-support module (not a test itself) holding plain-torch float32 reference
# implementations of the DSA (sparse attention with a learned top-k indexer) kernels, used by the
# sibling DSA tests as ground truth. It provides a reference for the indexer logits, a reference for
# sparse attention with optional attention sink, and a relative-difference metric. All references
# are differentiable through autograd and do no I/O.
"""fp32 torch references for the DSA kernels. Inputs follow miles.kernels.attention.dsa layouts."""

import torch


def indexer_logits_ref(q, k, weights, cu_seqlen_ks, cu_seqlen_ke):
    # [FUNCTION DOC] HELPER: Reference indexer logits for packed sequences. Inputs: queries of shape
    # tokens x heads x dim, keys of shape kv_tokens x dim (shared across heads), per-head weights of
    # shape tokens x heads, and per-query integer start (inclusive) and end (exclusive) key positions.
    # Computes in float32: per-head dot product of query and key, rectified linear unit, multiplied by
    # the head weight, then summed over heads. Returns a tokens x kv_tokens float matrix where key
    # positions outside each query's [start, end) range are set to negative infinity. Pure function;
    # differentiable w.r.t. q, k and weights.
    """Packed: q [T, H, D], k [T_kv, D], weights [T, H] -> [T, T_kv], -inf outside [ks, ke)."""
    scores = torch.einsum("thd,kd->thk", q.float(), k.float())
    scores = torch.relu(scores) * weights.float().unsqueeze(-1)
    logits = scores.sum(dim=1)
    positions = torch.arange(k.shape[0], device=q.device)
    valid = (positions >= cu_seqlen_ks.unsqueeze(1)) & (positions < cu_seqlen_ke.unsqueeze(1))
    return logits.masked_fill(~valid, float("-inf"))


def sparse_attention_ref(q, kv, indices, sm_scale, d_v, attn_sink=None):
    # [FUNCTION DOC] HELPER: Reference sparse attention over a per-query set of selected keys.
    # Inputs: q (batch, seq, heads, dim), kv (batch, kv_seq, groups, dim) used as both keys and values,
    # indices (batch, seq, groups, topk) of selected key positions with -1 meaning padding, softmax
    # scale, value width d_v (values are the first d_v channels of kv), and an optional per-head
    # attention sink logit. Heads are split evenly into groups (heads per group = heads / groups).
    # Per group: builds a boolean selection mask by accumulating hit counts per key (so a padded slot,
    # clamped to key 0, cannot erase a genuine selection of key 0); computes scaled scores; masks
    # unselected keys to negative infinity; subtracts a detached row max (including the sink if given;
    # a non-finite max is replaced by zero); exponentiates; the denominator is the weight sum plus the
    # exponentiated sink when present; output is weighted values divided by the denominator, with a
    # zero denominator replaced by one (so rows with nothing selected and no sink give zeros).
    # Returns float32 output (batch, seq, heads, d_v), concatenated over groups. Differentiable.
    """q [B, S, H, Dq], kv [B, S_kv, G, Dq], indices [B, S, G, topk] (-1 = padding), attn_sink [H] or None.
    Returns out [B, S, H, d_v]."""
    q, kv = q.float(), kv.float()
    batch, seq_len, heads, _ = q.shape
    seq_len_kv, groups = kv.shape[1], kv.shape[2]
    heads_per_group = heads // groups
    outs = []
    for g in range(groups):
        q_g = q[:, :, g * heads_per_group : (g + 1) * heads_per_group]
        kv_g = kv[:, :, g]
        idx = indices[:, :, g].long()
        valid = idx != -1
        # Padded slots clamp to key 0, so accumulate hits instead of scattering booleans: a plain
        # scatter of `valid` would let a padded slot overwrite a real selection of key 0.
        hits = torch.zeros(batch, seq_len, seq_len_kv, dtype=torch.int32, device=q.device)
        selected = hits.scatter_add_(-1, idx.clamp(min=0), valid.int()) > 0
        scores = torch.einsum("bshd,bkd->bshk", q_g, kv_g) * sm_scale
        scores = scores.masked_fill(~selected.unsqueeze(2), float("-inf"))
        row_max = scores.amax(dim=-1, keepdim=True)
        if attn_sink is not None:
            sink_g = attn_sink.float()[g * heads_per_group : (g + 1) * heads_per_group].view(1, 1, -1, 1)
            row_max = torch.maximum(row_max, sink_g)
        row_max = torch.where(torch.isinf(row_max), 0.0, row_max).detach()
        weights = torch.exp(scores - row_max)
        denom = weights.sum(dim=-1)
        if attn_sink is not None:
            denom = denom + torch.exp(sink_g - row_max).squeeze(-1)
        numer = torch.einsum("bshk,bkd->bshd", weights, kv_g[..., :d_v])
        outs.append(numer / torch.where(denom > 0, denom, 1.0).unsqueeze(-1))
    return torch.cat(outs, dim=2)


def rel_diff(a, b):
    # [FUNCTION DOC] HELPER: Returns a Python float: one minus twice the inner product of the two
    # flattened float32 tensors over the sum of their squared norms (zero for identical tensors).
    # Returns 0.0 when both tensors are all zeros. Tensors must have equal element counts.
    """Cosine-distance-like metric used by the miles dumper comparator."""
    x, y = a.flatten().float(), b.flatten().float()
    denom = (x * x).sum() + (y * y).sum()
    if denom == 0:
        return 0.0
    return (1.0 - 2.0 * (x * y).sum() / denom).item()
