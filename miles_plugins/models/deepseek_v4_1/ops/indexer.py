# [MODULE DOC] Sparse-attention indexer for DeepSeek-V4.1. Scores each query against compressed
# index keys with a weighted multi-head logit kernel and selects top-k picks, optionally restricted to
# a set of candidate blocks: the "candidate source" layer computes block-level candidates (highest
# block maxima plus each query's most recent block), and later layers reuse them as a mask. Has an
# optional fast selection path (external deep_select package, only on GPUs of compute capability 10 or
# higher, switchable by environment variable MILES_DSV41_DEEP_SELECT, default on) and a generic
# top-k path. Queries are processed in chunks (environment variable MILES_DSV41_INDEXER_QUERY_CHUNK,
# default 8192) to bound memory. Everything runs without gradients.
import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.kernels.attention.dsa import indexer_logits_sbhd
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn
from miles.kernels.position.rope import apply_rotary_emb
from miles.kernels.quant.fake_quant import fake_quant_fp4
from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.deepseek_v4.ops.norm import RMSNorm


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds a boolean mask of candidate keys. Pads the key axis with negative
    # infinity to a multiple of block size, takes the max logit per block, forces the block holding
    # each query's last visible key (index of compress length minus one, integer-divided by block size)
    # to positive infinity so it is always chosen, selects the top blocks (at most the requested count,
    # at most the number of blocks), keeps only selected blocks whose score is not negative infinity,
    # and expands the block mask back to key resolution, trimmed to the original width. Returns a
    # boolean tensor shaped like the logits. The compress lengths must be broadcastable against the
    # block axis (one per query, with a trailing singleton).
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


INDEXER_QUERY_CHUNK = int(os.environ.get("MILES_DSV41_INDEXER_QUERY_CHUNK", "8192"))

try:
    import deep_select
except ImportError:  # the torch path stays for images without it
    deep_select = None


@functools.lru_cache(maxsize=1)
def use_deep_select() -> bool:
    # [FUNCTION DOC] HELPER: Cached (computed once per process) decision whether to use the fast
    # selection package: false if it is not importable or the environment variable disables it (any
    # value other than "1"); otherwise true only when the current GPU's major compute capability is at
    # least 10. Queries the CUDA device on first call.
    if deep_select is None or os.environ.get("MILES_DSV41_DEEP_SELECT", "1") != "1":
        return False
    # the package carries sm100a and sm103a cubins, so anything older keeps torch.topk
    return torch.cuda.get_device_capability()[0] >= 10


def indexer_select(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    candidates: torch.Tensor | None,
    *,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    topk: int,
    topk_fn,
    allow_deep_select: bool,
    query_chunk: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # [FUNCTION DOC] HELPER: Inputs: queries (batch, sequence, heads, head dim), index keys (batch,
    # keys, head dim), per-head weights (batch, sequence, heads), per-query number of visible keys
    # (compress lengths, one per sequence position, shared across batch), optional candidate mask,
    # and flags/sizes. Keys are transposed to key-major; with the fast path, the key count is padded
    # up to the kernel's required row alignment. Queries are processed in chunks (whole sequence if the
    # chunk size is unset or large). Per chunk: compute logits with the logit kernel with key range
    # zero to the visible length (outside-range entries are negative infinity). If this is the
    # candidate source, compute the candidate mask from the logits; else if it uses candidates,
    # asserts a mask is given and masks out non-candidate keys in place. Then select: fast path takes
    # the top picks bounded by each row's visible end, fills out-of-bound slots with -1, pads to the
    # full top-k width with -1 when keys are fewer than top-k, and sorts indices ascending per row (so
    # -1 comes first); generic path pads logits with negative infinity to top-k width, calls the given
    # top-k function, sorts ascending, converts to 64-bit and replaces picks at or beyond the visible
    # length with -1. Returns (picks of shape batch, sequence, top-k as 64-bit integers; candidate
    # mask concatenated over chunks if this layer is the candidate source, otherwise the input mask
    # unchanged).
    # KNOWN ISSUE: the placement of invalid (-1) picks differs between paths: the fast path sorts
    # after filling, so -1 entries come first in each row, while the generic path sorts before
    # replacing out-of-range picks, so -1 entries come last. Consumers must not rely on position.
    bsz, seqlen = q.shape[:2]
    n_kv = index_k.size(1)
    k_t = index_k.transpose(0, 1).contiguous()
    deep_selecting = allow_deep_select and use_deep_select()
    if deep_selecting:
        # deep_select requires aligned score rows; the padded columns sit past every query's length
        align = deep_select.get_stride_requirement()[0] // 4
        pad_kv = -(-n_kv // align) * align - n_kv
        if pad_kv:
            k_t = torch.nn.functional.pad(k_t, (0, 0, 0, 0, 0, pad_kv))
    chunk = seqlen if not query_chunk or query_chunk >= seqlen else query_chunk
    idx_parts: list[torch.Tensor] = []
    cand_parts: list[torch.Tensor] = []
    for s in range(0, seqlen, chunk):
        e = min(s + chunk, seqlen)
        lens = compress_lens[s:e]
        scores = indexer_logits_sbhd(
            q[:, s:e].transpose(0, 1).contiguous(),
            k_t,
            weights[:, s:e].transpose(0, 1).float().contiguous(),
            torch.zeros(e - s, dtype=torch.int32, device=q.device),
            lens.to(torch.int32),
        )
        # indexer_logits_sbhd already wrote -inf outside [0, lens) for every query
        if is_candidate_source:
            cand = select_candidate_blocks(
                scores[..., :n_kv], lens.unsqueeze(-1), candidate_topk_blocks, candidate_block_size
            )
            cand_parts.append(cand)
        elif uses_candidates:
            assert candidates is not None
            scores[..., :n_kv].masked_fill_(~candidates[:, s:e], -torch.inf)
        if deep_selecting:
            flat = scores.reshape(bsz * (e - s), scores.size(-1))
            row_end = lens.to(torch.int32).repeat(bsz)
            _, idx = deep_select.topk(
                flat,
                min(topk, n_kv),
                end=row_end,
                indices_type=torch.int64,
                return_value=False,
                idx_oob_fill_value=-1,
            )
            idx = idx.reshape(bsz, e - s, -1)
            if n_kv < topk:
                idx = torch.nn.functional.pad(idx, (0, topk - n_kv), value=-1)
            idx_parts.append(idx.sort(dim=-1).values)
            continue
        if n_kv < topk:
            scores = torch.nn.functional.pad(scores, (0, topk - n_kv), value=-torch.inf)
        idx = topk_fn(scores.reshape(bsz * (e - s), scores.size(-1)), topk)
        idx = idx.reshape(bsz, e - s, topk).sort(dim=-1).values.to(torch.int64)
        idx_parts.append(torch.where(idx < lens.unsqueeze(-1), idx, -1))
    idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=1)
    if is_candidate_source:
        candidates = cand_parts[0] if len(cand_parts) == 1 else torch.cat(cand_parts, dim=1)
    return idx, candidates


class DeepSeekV41Indexer(MegatronModule):
    # [CLASS DOC] Megatron module with duplicated (not tensor-parallel) projections producing index
    # queries (heads times index head dim) from the low-rank query and per-head weights from hidden
    # states; if it owns keys (is a key/value source layer) also a key projection from the latent to
    # index head dim plus norm. Flags say whether it is the candidate source and whether it consumes
    # candidates. Registers with the indexer replay manager.
    def __init__(
        self,
        config: TransformerConfig,
        layer_id: int,
        head_dim: int,
        compress_ratio: int,
        owns_k: bool,
        is_candidate_source: bool,
        uses_candidates: bool,
    ):
        # [FUNCTION DOC] Reads candidate top-k block count and block size (v41 config fields), indexer
        # head count, head dim, top-k, rope head dim from the config; the softmax scale is index head
        # dim to the minus one half. Query low-rank rank falls back to hidden size when unset. Builds
        # the two bias-free duplicated linears; if owns_k, builds a bf16 key projection and a norm with
        # fp32 scale. Registers with the replay manager using the layer's rank among the sorted index
        # source layers as stream index (raises a value error if the layer is not an index source).
        # The head_dim argument is the latent width; compress_ratio is stored but not otherwise used.
        super().__init__(config=config)
        self.layer_id = layer_id
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.is_candidate_source = is_candidate_source
        self.uses_candidates = uses_candidates
        self.candidate_topk_blocks = config.v41_candidate_topk_blocks
        self.candidate_block_size = config.v41_candidate_block_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.softmax_scale = self.index_head_dim**-0.5
        q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size

        self.linear_wq_b = TELinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_weights_proj = TELinear(
            config.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        if owns_k:
            self.linear_wk = nn.Linear(head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, config.layernorm_epsilon)
            mark_keep_in_fp32(self.k_norm.weight)
        indexer_replay_manager.register_to_module(
            self, "indexer_replay", stream_idx=sorted(config.v41_index_source_layer_ids).index(layer_id)
        )

    @torch.no_grad()
    def index_keys(self, latent: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] No-gradient. Projects the compressed latent (batch, groups, latent dim) to
        # index head dim, normalizes, copies, applies rotary in place on the trailing rope dims with
        # the given per-group frequencies, and fake-quantizes to fp4. Returns index keys (batch,
        # groups, index head dim). Only valid on modules that own keys.
        k = self.k_norm(F.linear(latent, self.linear_wk.weight)).clone()
        apply_rotary_emb(k[..., -self.rope_head_dim :], freqs_cis)
        return fake_quant_fp4(k)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        index_k: torch.Tensor,
        freqs_cis: torch.Tensor,
        compress_lens: torch.Tensor,
        candidates: torch.Tensor | None,
    ):
        # [FUNCTION DOC] No-gradient. Inputs: hidden states (batch, sequence, hidden), low-rank query,
        # index keys, rotary frequencies for queries, per-query visible compressed count, and optional
        # candidate mask. Projects queries to heads, copies, rotates the trailing rope dims in place,
        # fake-quantizes to fp4; weights are the projection scaled by softmax scale times head count to
        # the minus half. Builds the top-k function from the replay manager wrapped around the generic
        # torch top-k backend (the backend is hard coded, not taken from a config). The fast path is
        # allowed unless replay is enabled and not in its fallthrough stage. Delegates to the select
        # helper with the configured top-k and chunk size. Returns (picks, candidates).
        bsz, seqlen, _ = x.shape
        rd = self.rope_head_dim
        q, _ = self.linear_wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim).clone()
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = fake_quant_fp4(q)
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.softmax_scale * self.index_n_heads**-0.5)

        topk_fn = indexer_replay_manager.get_topk_fn(get_dsa_topk_fn("torch"), return_probs=False)
        idx, candidates = indexer_select(
            q,
            index_k,
            weights,
            compress_lens,
            candidates,
            is_candidate_source=self.is_candidate_source,
            uses_candidates=self.uses_candidates,
            candidate_topk_blocks=self.candidate_topk_blocks,
            candidate_block_size=self.candidate_block_size,
            topk=self.index_topk,
            topk_fn=topk_fn,
            allow_deep_select=not indexer_replay_manager.enabled or indexer_replay_manager.stage == "fallthrough",
            query_chunk=INDEXER_QUERY_CHUNK,
        )
        return idx, candidates
