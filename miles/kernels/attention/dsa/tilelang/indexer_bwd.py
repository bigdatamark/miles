# [MODULE DOC] Tilelang backward kernel of the DSA lightning indexer for the selected top-k keys.
# The forward score of query t and key j is the sum over heads h of weight[t,h] times relu(q[t,h]
# dot k[j]). Given the gradient of the scores at the selected keys (queries by top-k), the kernel
# recomputes, per selected key, the relu'd per-head dot products and produces: gradient of the
# queries (g times weight times indicator(dot > 0) times k, summed over the selected keys),
# gradient of the weights (g times relu(dot), summed over the selected keys), and gradient of the
# keys (g times weight times indicator times q, summed over heads, scattered with atomic addition
# into rows of the key gradient). Host entry point: indexer_bwd. Requires the tilelang package.
# ruff: noqa
# Adapted from https://github.com/tile-ai/tilelang/blob/4956b5835fa554af6c03d4a6289cad44bf310869/examples/dsa_sparse_finetune/indexer_bwd.py
import tilelang as tl
import tilelang.language as T
import torch

BF16 = T.bfloat16
FP32 = T.float32
INT32 = T.int32

pass_configs = {
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tl.jit(pass_configs=pass_configs)
def _indexer_bwd_kernel(
    heads: int,
    dim: int,
    topk: int,
    block_I: int = 32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    # [FUNCTION DOC] HELPER: Kernel factory (compiled and cached by tilelang's jit on the argument
    # values). Builds the backward kernel for a given head count, head dim and top-k. Asserts: the
    # pipeline stage count is 0 (the pipelined loop is disabled), top-k is a power of two and a
    # multiple of block_I (32 selected keys per iteration), and heads is at most 64 and a multiple
    # of 8. Query/key/gradient shared tiles use float32; head tiles are padded to at least 16 rows
    # with zeros. Uses TMA-lowering and warp-specialization disabled. Returns the program below.
    assert num_stages == 0
    assert topk == tl.math.next_power_of_2(topk)
    assert topk % block_I == 0
    assert heads <= 64 and heads % 8 == 0
    seq_len = T.symbolic("seq_len")
    q_seq_len = T.symbolic("q_seq_len")

    dtype: str = BF16
    accum_dtype: str = FP32
    index_q_shape = [q_seq_len, heads, dim]
    weights_shape = [q_seq_len, heads]
    index_k_shape = [seq_len, dim]
    shape_p = [q_seq_len, topk]
    topk_indices_shape = [q_seq_len, topk]

    pad_heads = heads
    if heads < 16:
        pad_heads = 16

    @T.prim_func
    def indexer_bwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexK: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        OGrad: T.Tensor(shape_p, FP32),
        dIndexQ: T.Tensor(index_q_shape, dtype),
        dWeights: T.Tensor(weights_shape, FP32),
        dIndexK: T.Tensor(index_k_shape, FP32),
    ):
        # [FUNCTION DOC] HELPER: The device program; one thread block per query (grid of query
        # count). Inputs: queries (queries by heads by dim, bfloat16), keys (kv length by dim,
        # bfloat16), weights (queries by heads, float32), selected key indices (queries by top-k,
        # 32-bit, minus one or out-of-range meaning padding), and the score gradient (queries by
        # top-k, float32). Outputs: dIndexQ (bfloat16), dWeights (float32) written once per query,
        # and dIndexK (float32) accumulated with atomic adds, so it must be zero-initialized.
        # Steps: load the query's head matrix and weights into shared memory; for each block of 32
        # selected keys: load indices and gradients, gather key rows (zero rows for invalid
        # indices), compute dot products (keys by heads), apply relu, accumulate the weight
        # gradient as gradient times relu'd dot summed over the block, build a masked matrix
        # gradient times weight where the dot is positive (zero for padded heads), accumulate the
        # query gradient from that matrix and the keys, compute per-key gradients from that matrix
        # and the query head matrix and atomically add them into the key gradient rows of valid
        # indices. Finally writes the first `heads` rows of the accumulators.
        # Invalid (padded) selected keys contribute zero because their gathered key is zero.
        with T.Kernel(q_seq_len, threads=num_threads) as (bx):
            index_q_shared = T.alloc_shared([pad_heads, dim], dtype=FP32)
            weights_shared = T.alloc_shared([pad_heads], dtype=FP32)
            index_k_shared = T.alloc_shared([block_I, dim], dtype=FP32)
            indices_shared = T.alloc_shared([block_I], dtype=INT32)
            d_index_q_frag = T.alloc_fragment([pad_heads, dim], dtype=accum_dtype)
            d_weights_frag = T.alloc_fragment([pad_heads], dtype=accum_dtype)
            d_index_k_frag = T.alloc_fragment([block_I, dim], dtype=accum_dtype)
            logits = T.alloc_fragment((block_I, pad_heads), dtype=accum_dtype)
            _logits = T.alloc_shared((block_I, pad_heads), dtype=accum_dtype)
            grad = T.alloc_shared([block_I], dtype=FP32)

            num_blocks = T.ceildiv(topk, block_I)
            for i, j in T.Parallel(pad_heads, dim):
                index_q_shared[i, j] = T.if_then_else(i < heads, IndexQ[bx, i, j], 0)
            for i in T.Parallel(heads):
                weights_shared[i] = Weights[bx, i]

            T.fill(d_index_q_frag, 0)
            T.fill(d_weights_frag, 0)

            # for bi_i in T.Pipelined(num_blocks, num_stages=num_stages):
            for bi_i in T.serial(num_blocks):
                for i in T.Parallel(block_I):
                    if bi_i * block_I + i < topk:
                        indices_shared[i] = TopkIndices[bx, bi_i * block_I + i]
                        grad[i] = OGrad[bx, bi_i * block_I + i]

                T.sync_threads()
                for i, j in T.Parallel(block_I, dim):
                    index_k_shared[i, j] = T.if_then_else(
                        indices_shared[i] > -1 and indices_shared[i] < seq_len, IndexK[indices_shared[i], j], 0
                    )

                T.sync_threads()
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                for i, j in T.Parallel(block_I, heads):
                    logits[i, j] = T.max(logits[i, j], 0)

                d_weights_i = T.alloc_fragment((block_I, pad_heads), accum_dtype)
                for i, j in T.Parallel(block_I, heads):
                    d_weights_i[i, j] = grad[i] * logits[i, j]
                T.reduce_sum(d_weights_i, d_weights_frag, dim=0, clear=False)

                for i, j in T.Parallel(block_I, pad_heads):
                    _logits[i, j] = T.if_then_else(logits[i, j] > 0 and j < heads, grad[i] * weights_shared[j], 0)
                T.sync_threads()
                T.gemm(
                    _logits,
                    index_k_shared,
                    d_index_q_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )

                T.gemm(
                    _logits,
                    index_q_shared,
                    d_index_k_frag,
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=True,
                )

                for i, j in T.Parallel(block_I, dim):
                    if indices_shared[i] > -1 and indices_shared[i] < seq_len:
                        T.atomic_add(dIndexK[indices_shared[i], j], d_index_k_frag[i, j])

            T.copy(d_index_q_frag[:heads, :], dIndexQ[bx, :, :])
            T.copy(d_weights_frag[:heads], dWeights[bx, :])

    return indexer_bwd_kernel


def indexer_bwd(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
):
    # [FUNCTION DOC] Host wrapper. Inputs: index_q (queries by heads by dim, bfloat16), weights
    # (queries by heads, float32), index_k (kv length by dim, bfloat16), topk_indices (queries by
    # top-k, 32-bit; top-k must be a power of two multiple of 32, which the caller pads to), and
    # grad_scores (queries by top-k, must be float32 to match the kernel signature). Allocates
    # outputs: grad_q like index_q (uninitialized, fully written), grad_w float32 like weights,
    # grad_k float32 zeros like index_k; makes inputs contiguous and launches the kernel compiled
    # for (heads, dim, top-k). Returns (grad_q, grad_w, grad_k) in that order.
    # KNOWN ISSUE: the squeeze of the last dimension applied to the weights and to grad_w is a no-op
    # for every supported head count (the kernel requires at least 8 heads, so the last dimension
    # is never of size 1); it looks like a leftover.
    _, head_num, head_dim = index_q.shape
    k_top = topk_indices.shape[1]

    grad_scores = grad_scores.contiguous()
    grad_q = torch.empty_like(index_q)
    grad_w = torch.empty_like(weights, dtype=torch.float32)
    grad_k = torch.zeros_like(index_k, dtype=torch.float32)

    _indexer_bwd_kernel(head_num, head_dim, k_top)(
        index_q.contiguous(),
        index_k.contiguous(),
        weights.squeeze(-1).contiguous(),
        topk_indices.contiguous(),
        grad_scores,
        grad_q,
        grad_w.squeeze(-1),
        grad_k,
    )

    return grad_q, grad_w, grad_k
