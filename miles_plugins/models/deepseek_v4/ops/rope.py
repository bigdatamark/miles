# [MODULE DOC] HELPER: Rotary position embedding for DeepSeek-V4. Builds (and caches) complex
# rotation tables, optionally with YaRN long-context frequency interpolation, applies them in place
# to the trailing channels of activations (interpreting adjacent channel pairs as complex numbers),
# and provides a config-driven wrapper that asserts the supported DeepSeek-V4 rope settings.
import math
from functools import lru_cache

import torch
from megatron.core.transformer import TransformerConfig


@lru_cache(maxsize=32)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device=None
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Memoized (32 entries; arguments must be hashable, device included) table
    # builder. Base inverse frequencies are base to the power of minus (even index over dim), giving
    # dim/2 values. If original_seq_len is positive (YaRN): compute low and high correction
    # dimensions from beta_fast and beta_slow (clamped into range zero to dim-1), build a linear ramp
    # over the dim/2 frequencies between them, and blend each frequency between its value divided by
    # factor (interpolated) and its original value (extrapolated) using one minus the ramp as the
    # smoothing weight. Then takes the outer product of positions 0..seqlen-1 with the frequencies and
    # converts to unit-magnitude complex numbers. Built on CPU in float32 and moved to the device when
    # given. Returns a complex tensor of shape seqlen by dim/2. Inner helpers: correction dimension
    # (maps a rotation count to a dimension index), correction range (floor/ceil, clamped), and
    # linear ramp (clamped zero to one; widens a degenerate range by a small epsilon).
    # KNOWN ISSUE: the blending weight is named smooth but is already one minus the ramp, then used as
    # one minus smooth for the interpolated term, which is the standard YaRN blend; the docstring's
    # description is consistent. The cache holds the table keyed by device so memory stays resident.
    """Precompute the complex rotary frequencies for RoPE, with optional YaRN smoothing.

    When ``original_seq_len > 0``, applies YaRN factor rescaling interpolated
    by a linear ramp between ``beta_fast`` and ``beta_slow``. Otherwise the
    base frequencies are used verbatim.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        # [FUNCTION DOC] HELPER: YaRN formula: dimension index at which the rotary frequency completes
        # the given number of rotations over max_seq_len positions.
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        # [FUNCTION DOC] HELPER: Returns the pair (floor of the correction dim for low_rot clamped to
        # at least zero, ceiling of the correction dim for high_rot clamped to at most dim minus one).
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        # [FUNCTION DOC] HELPER: Float32 vector of length dim whose entry i is (i minus min) over
        # (max minus min), clamped to the range zero to one. If min equals max, max is nudged up by
        # 0.001 to avoid division by zero.
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # Cache on `device` so repeated lengths are GPU cache hits; values bit-identical to CPU build.
    return freqs_cis.to(device) if device is not None else freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Upcasts x to float32, views adjacent channel pairs as complex numbers,
    # multiplies by the table (conjugated when inverse), flattens back to real and copies the result
    # into x in place (casting back to x's dtype, so the caller's tensor, including a slice view, is
    # mutated); also returns x. The table is reshaped by rank: for 3-dimensional x (batch, positions,
    # channels) to broadcast as 1, positions, pairs; otherwise assumed 4-dimensional (batch,
    # positions, heads, channels) as 1, positions, 1, pairs. The table's entry count must equal the
    # position axis (dim 1) of x. Channel count must be even.
    """Apply RoPE in-place to the last dim of ``x``.

    ``x`` has shape ``[..., dim]`` where ``dim`` is even; the last-dim pairs are
    treated as complex numbers multiplied by ``freqs_cis``. When ``inverse=True``
    the conjugate rotation is applied (used for the indexer's inverse rope).
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def wrapped_precompute_freqs_cis(
    config: TransformerConfig, rope_head_dim: int, base: float, yarn_disabled: bool, max_seq_len: int, device
):
    # [FUNCTION DOC] HELPER: Builds the rotation table for the given length from config values:
    # original context length (zero if yarn is disabled, else config's original max position
    # embeddings), scaling factor, and beta fast/slow. Asserts the scaling factor is 4 or 16 and that
    # the effective parameters equal the only supported ones (original length 65536 or 0 when yarn
    # is disabled, beta fast 32, beta slow 1); any other config trips an assertion. Returns the cached
    # table from the precompute function on the given device.
    # max_seq_len = global length (cp_size * seqlen_local); table rebuilt per call, lru-cached.
    # yarn_disabled=True → original_seq_len=0, which makes precompute_freqs_cis skip the YaRN
    # correction-range interpolation. Used by 0415 for pure-window (compress_ratio==0) layers.
    original_seq_len = 0 if yarn_disabled else config.original_max_position_embeddings

    inputs = dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=original_seq_len,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
    )

    assert config.rotary_scaling_factor in (4, 16), f"Unexpected rotary_scaling_factor: {config.rotary_scaling_factor}"
    expected_original = 0 if yarn_disabled else 65536
    assert inputs == dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=expected_original,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=32,
        beta_slow=1,
    )

    return precompute_freqs_cis(**inputs, device=device)
