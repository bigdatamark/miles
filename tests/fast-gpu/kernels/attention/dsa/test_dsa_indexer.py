# [MODULE DOC] GPU tests for the DSA lightning-indexer kernels (packed and batched indexer logits,
# top-k score gradients, and the combined indexer plus top-k wrapper) against the float32 reference in
# the sibling dsa_reference module, which is loaded by file path. Registered for CI on a 2-GPU Hopper
# suite with the precision label. The module is skipped when tilelang is missing or CUDA is
# unavailable. Run directly, it invokes pytest on itself.
import importlib.util
import pathlib

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

tilelang = pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa import (  # noqa: E402
    causal_ranges,
    causal_ranges_compressed,
    get_dsa_topk_fn,
    indexer_logits,
    indexer_logits_sbhd,
    indexer_topk_scores,
    lighting_indexer,
)

_spec = importlib.util.spec_from_file_location("dsa_reference", pathlib.Path(__file__).with_name("dsa_reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def _packed_inputs(cu_seqlens, heads, dim):
    # [FUNCTION DOC] HELPER: Builds random packed indexer inputs on the GPU from a list of cumulative
    # sequence lengths (last entry is the total token count). Returns queries (total, heads, dim) and
    # keys (total, dim) in bfloat16, float32 weights (total, heads) scaled by 0.01, and integer
    # causal start/end key ranges per token from the causal ranges helper. Uses the global random state.
    total = cu_seqlens[-1]
    q = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(total, heads, device="cuda", dtype=torch.float32) * 0.01
    ks, ke = causal_ranges(torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32))
    return q, k, weights, ks.int(), ke.int()


@pytest.mark.parametrize("cu_seqlens", [[0, 128], [0, 100, 356], [0, 64, 96, 1024]])
@pytest.mark.parametrize("heads", [8, 64])
def test_indexer_logits_packed_matches_reference(cu_seqlens, heads):
    # [FUNCTION DOC] Parametrized over three packings (single, two and three sequences) and head
    # counts 8 and 64, with dim 128 and seed 0. Asserts the kernel logits have negative infinity in
    # exactly the same positions as the reference, and that on the finite entries the relative
    # difference is below 1e-3.
    torch.manual_seed(0)
    q, k, weights, ks, ke = _packed_inputs(cu_seqlens, heads, 128)
    ref = reference.indexer_logits_ref(q, k, weights, ks, ke)
    out = indexer_logits(q, k, weights, ks, ke)
    assert torch.equal(torch.isinf(out), torch.isinf(ref))
    finite = ~torch.isinf(ref)
    assert reference.rel_diff(out[finite], ref[finite]) < 1e-3


@pytest.mark.parametrize("seqlen,batch,compress_ratio", [(128, 1, 4), (512, 2, 4), (2048, 1, 128)])
def test_indexer_logits_sbhd_matches_reference(seqlen, batch, compress_ratio):
    # [FUNCTION DOC] Parametrized over (seqlen, batch, compress ratio). Builds sequence-first
    # batched inputs with 16 heads and dim 128, where keys have seqlen divided by the compression
    # ratio positions; ranges come from the compressed causal ranges helper. For each batch element,
    # compares the batched kernel output to the reference: infinity patterns must match exactly and the
    # relative difference on finite entries must be below 1e-3.
    torch.manual_seed(0)
    heads, dim = 16, 128
    seqlen_kv = seqlen // compress_ratio
    q = torch.randn(seqlen, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seqlen_kv, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seqlen, batch, heads, device="cuda", dtype=torch.float32) * 0.01
    ks, ke = causal_ranges_compressed(seqlen, compress_ratio, q.device)
    out = indexer_logits_sbhd(q, k, weights, ks, ke)
    for b in range(batch):
        ref = reference.indexer_logits_ref(q[:, b], k[:, b], weights[:, b], ks, ke)
        assert torch.equal(torch.isinf(out[b]), torch.isinf(ref))
        finite = ~torch.isinf(ref)
        assert reference.rel_diff(out[b][finite], ref[finite]) < 1e-3


@pytest.mark.parametrize("topk", [32, 100, 512])
def test_indexer_topk_scores_backward_matches_autograd(topk):
    # [FUNCTION DOC] Parametrized over top-k 32, 100, 512. Uses packed inputs with sequences of 300
    # and 724 tokens, 16 heads. Takes the top-k key positions of the kernel logits (clamped to the key
    # count), marking entries whose logit is negative infinity as padding (-1). Computes reference
    # gradients by autograd through the float32 reference logits gathered at those indices, and kernel
    # gradients through the top-k score function; both losses are the sum of valid scores times the same
    # random upstream gradient. Asserts the gradients w.r.t. queries, keys and weights agree with
    # relative difference below 1e-4.
    torch.manual_seed(0)
    q, k, weights, ks, ke = _packed_inputs([0, 300, 1024], 16, 128)
    logits = indexer_logits(q, k, weights, ks, ke)
    topk_indices = logits.topk(min(topk, logits.shape[-1]), dim=-1).indices.int()
    topk_indices = topk_indices.masked_fill(torch.gather(logits, -1, topk_indices.long()) == -torch.inf, -1)

    q_ref = q.clone().float().requires_grad_()
    k_ref = k.clone().float().requires_grad_()
    w_ref = weights.clone().requires_grad_()
    ref_logits = reference.indexer_logits_ref(q_ref, k_ref, w_ref, ks, ke)
    ref_scores = torch.gather(ref_logits, -1, topk_indices.clamp(min=0).long())
    grad_scores = torch.randn(topk_indices.shape, device="cuda", dtype=torch.float32)
    (torch.where(topk_indices != -1, ref_scores, 0.0) * grad_scores).sum().backward()

    q_tl = q.clone().requires_grad_()
    k_tl = k.clone().requires_grad_()
    w_tl = weights.clone().requires_grad_()
    scores = indexer_topk_scores(q_tl, k_tl, w_tl, logits, topk_indices)
    (torch.where(topk_indices != -1, scores, 0.0) * grad_scores).sum().backward()

    for ref_g, tl_g in ((q_ref.grad, q_tl.grad), (k_ref.grad, k_tl.grad), (w_ref.grad, w_tl.grad)):
        assert reference.rel_diff(ref_g, tl_g) < 1e-4


def test_lighting_indexer_returns_topk_of_its_own_logits():
    # [FUNCTION DOC] Runs the combined indexer on one 512-token sequence (8 heads, top-k 64, torch
    # top-k backend). Asserts scores and indices both have shape 512 by 64, that returned scores for
    # valid indices equal the separately computed logits gathered at those indices exactly, that the
    # first query (one valid key) has 63 padded slots, and that all padded slots carry negative
    # infinity scores.
    torch.manual_seed(0)
    q, k, weights, ks, ke = _packed_inputs([0, 512], 8, 128)
    scores, topk_indices = lighting_indexer(q, k, weights, ks, ke, topk=64, topk_fn=get_dsa_topk_fn("torch"))
    logits = indexer_logits(q, k, weights, ks, ke)
    assert scores.shape == topk_indices.shape == (512, 64)
    valid = topk_indices != -1
    gathered = torch.gather(logits, -1, topk_indices.clamp(min=0).long())
    assert torch.equal(scores[valid], gathered[valid])
    # early queries have fewer than 64 valid keys, so padding must be -1 with -inf scores
    assert (topk_indices[0] == -1).sum() == 63
    assert torch.all(scores[~valid] == -torch.inf)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
