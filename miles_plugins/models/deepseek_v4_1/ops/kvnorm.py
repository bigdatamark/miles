# [MODULE DOC] HELPER: Key/value preparation for DeepSeek-V4.1 that makes training numerics match the
# inference engine's fp8 paged key/value cache. Each function computes a differentiable "soft" result
# (norm, rotary, fp8 simulation) and a "hard" result (real round trip through the engine's cache kernels
# from the sglang package, no gradient), and returns the hard value with the soft value's gradient
# (straight-through trick). Paged cache layout constants: 64 tokens per page, 584 bytes per token,
# 576-byte rows.
import torch

from miles.kernels.position.rope import apply_rotary_emb
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat

_PAGE_SIZE = 64
_BYTES_PER_TOKEN = 584
_ROW_BYTES = 576


def kv_norm_rope(
    kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: In float32: RMS-normalize over the last dim (epsilon inside the root),
    # scale by the weight, make contiguous, apply rotary in place to the trailing rope-dim channels
    # with the given frequencies, and cast back to the input dtype. Differentiable.
    x = kv.float()
    data = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()).contiguous()
    apply_rotary_emb(data[..., -rope_dim:], freqs_cis)
    return data.to(kv.dtype)


@torch.no_grad()
def kv_cache_roundtrip(kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: No-gradient. Imports the engine kernels lazily (import error if sglang is
    # missing). Flattens (batch, sequence, dim) to tokens, positions restart at zero per batch row,
    # slot locations are consecutive. Allocates a zeroed byte cache with enough pages and rows aligned
    # to the row size, runs the engine's fused norm + rotary + store kernel (weight cast to the input
    # dtype), then dequantizes the same slots from the cache. Returns a tensor shaped like the input
    # in the input dtype. GPU kernels; allocates a temporary cache.
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla

    bsz, seqlen, dim = kv.shape
    flat = kv.reshape(bsz * seqlen, dim).contiguous()
    positions = torch.arange(seqlen, device=kv.device, dtype=torch.int32).repeat(bsz)
    loc = torch.arange(bsz * seqlen, device=kv.device, dtype=torch.int32)
    pages = (bsz * seqlen + _PAGE_SIZE) // _PAGE_SIZE + 1
    page_bytes = -(-(_BYTES_PER_TOKEN * _PAGE_SIZE) // _ROW_BYTES) * _ROW_BYTES
    cache = torch.zeros(pages, page_bytes, dtype=torch.uint8, device=kv.device)
    fused_k_norm_rope_flashmla(
        kv=flat,
        kv_weight=weight.to(kv.dtype).contiguous(),
        eps=eps,
        freqs_cis=freqs_cis.contiguous(),
        positions=positions,
        out_loc=loc,
        kvcache=cache,
        page_size=_PAGE_SIZE,
    )
    return dequantize_k_cache_paged(cache, loc, _PAGE_SIZE).reshape(bsz, seqlen, dim).to(kv.dtype)


def kv_norm_rope_fp8(
    kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    # [FUNCTION DOC] Soft path: norm and rotary, then fp8-simulate (block 64) all channels except the
    # trailing rope channels. Hard path: engine cache round trip of the same input. Returns the hard
    # values numerically, with gradient flowing through the soft path (soft plus detached difference).
    soft = kv_norm_rope(kv, weight, eps, freqs_cis, rope_dim).clone()
    soft[..., :-rope_dim] = fp8_simulate_qat(soft[..., :-rope_dim], 64)
    hard = kv_cache_roundtrip(kv, weight, eps, freqs_cis)
    return soft + (hard - soft).detach()


@torch.no_grad()
def compressed_kv_cache_roundtrip(latent: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: No-gradient. Like the key/value round trip but for already-processed
    # latents: stores them in a temporary byte cache with the engine's store kernel (flashmla layout,
    # 64-bit slot ids) and dequantizes them back. Returns a tensor shaped like the input in its dtype.
    from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged

    bsz, seqlen, dim = latent.shape
    flat = latent.reshape(bsz * seqlen, dim).contiguous()
    loc = torch.arange(bsz * seqlen, device=latent.device, dtype=torch.int64)
    pages = (bsz * seqlen + _PAGE_SIZE) // _PAGE_SIZE + 1
    page_bytes = -(-(_BYTES_PER_TOKEN * _PAGE_SIZE) // _ROW_BYTES) * _ROW_BYTES
    cache = torch.zeros(pages, page_bytes, dtype=torch.uint8, device=latent.device)
    fused_store_cache(input=flat, cache=cache, indices=loc, page_size=_PAGE_SIZE, type="flashmla")
    return dequantize_k_cache_paged(cache, loc, _PAGE_SIZE).reshape(bsz, seqlen, dim).to(latent.dtype)


def compressed_kv_stored(latent: torch.Tensor, rope_dim: int) -> torch.Tensor:
    # [FUNCTION DOC] Same straight-through scheme as the key/value version: soft path simulates fp8
    # (block 64) on all but the trailing rope channels of a copy; hard path is the cache round trip;
    # returns hard values with the soft gradient.
    """The compressed latent as the engine's fp8 pool returns it: the fp4/E4M3 values are requantized to fp8 per 64."""
    soft = latent.clone()
    soft[..., :-rope_dim] = fp8_simulate_qat(soft[..., :-rope_dim], 64)
    hard = compressed_kv_cache_roundtrip(latent)
    return soft + (hard - soft).detach()
