# [MODULE DOC] Tilelang forward kernels of the DSA lightning indexer. Score of query t against key
# j = sum over heads h of weights[t,h] times relu(q[t,h] dot k[j]), written as a dense float32
# matrix (queries by keys). Two kernels: the scoring kernel, which computes scores for the union of
# key ranges of a block of queries, and a cleaning kernel which overwrites every entry outside
# each query's own key range [start, end) with minus infinity. Host entry point: indexer_fwd.
# Requires the tilelang package and a GPU.
# ruff: noqa
# Adapted from https://github.com/tile-ai/tilelang/blob/4956b5835fa554af6c03d4a6289cad44bf310869/examples/deepseek_v32/fp8_lighting_indexer.py
import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def _indexer_fwd_kernel(
    heads,
    index_dim,
    block_N=128,
    num_stages=2,
    threads=512,
    block_Q=None,
):
    # [FUNCTION DOC] HELPER: Kernel factory (jit-compiled, cached per argument values) for the
    # scoring kernel with fast math enabled. Parameters: head count, head dim, keys per iteration
    # block_N (128), pipeline stages (2), threads (512), and queries per thread block block_Q
    # (default 256 integer-divided by heads, at least 1). Sequence and key lengths are dynamic.
    # Returns the program below.
    if block_Q is None:
        block_Q = max(1, 256 // heads)
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def indexer_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        # [FUNCTION DOC] HELPER: Device program; one thread block per block of block_Q queries.
        # Inputs: queries flattened to rows of (query, head) by dim, keys, per-query range start and
        # end (32-bit), per-head float32 weights; output: float32 scores (queries by keys). Steps:
        # reduce the block's range starts to their minimum and ends to their maximum (each clamped
        # to the key count); load the block's query rows and weights; then iterate (software
        # pipelined) over blocks of block_N keys covering that union range: matrix product of keys
        # with the block's queries, relu, multiply by the head weight, sum over heads, and store into
        # the output at the key's absolute column. Entries inside the union range but outside an
        # individual query's own range are still written with computed scores, and columns outside the
        # union range are never written, so callers must run the cleaning kernel to get exact
        # minus-infinity masking.
        # KNOWN ISSUE: there is no handling of a last query block that extends past the number of
        # queries (range arrays and queries are read for the full block), so the query count is
        # implicitly assumed to be a multiple of block_Q unless the compiler inserts guards.
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return indexer_fwd_kernel


@tilelang.jit
def _clean_logits_kernel(
    threads: int = 512,
    block_K: int = 4096,
):
    # [FUNCTION DOC] HELPER: Kernel factory (jit-compiled) for the cleaning kernel; threads per block
    # (512) and columns per tile (4096) are parameters. Returns the program below.
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        # [FUNCTION DOC] HELPER: Device program; one thread block per query row. Walks the row in tiles
        # of block_K columns with each thread handling every threads-th column, and in place sets each
        # entry whose column is before the query's range start or at/after its range end to minus
        # infinity. Entries inside the range are left unchanged.
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def indexer_fwd(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    # [FUNCTION DOC] Host wrapper. q is queries by heads by dim (bfloat16, must be viewable as
    # (queries times heads) by dim, i.e. contiguous), kv is keys by dim (bfloat16), weights is
    # queries by heads (float32), and the two range tensors are 32-bit per-query key range start and
    # end (end exclusive). Allocates an uninitialized float32 matrix of queries by keys, runs the
    # scoring kernel, and, if clean_logits (default true), runs the cleaning kernel so every entry
    # outside a query's range is minus infinity. With clean_logits false the entries outside the
    # scored union ranges are uninitialized memory. Returns the matrix.
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    clean_logits_kernel = _clean_logits_kernel()

    kernel = _indexer_fwd_kernel(heads=heads, index_dim=index_dim)

    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    kernel(
        q.view(seq_len * heads, index_dim),
        kv,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    if clean_logits:
        clean_logits_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits
