# [MODULE DOC]
# Registers an attention function named "triton" in the transformers attention registry that
# computes dense causal self-attention with SGLang's unified extend-attention Triton kernel in the
# forward pass and the project's dense Triton backward kernel in the backward pass, wrapped in a
# custom autograd function. Batches are treated as B equal-length sequences of length S with no
# padding mask, no prefix cache, and no dropout. The apply function registers it and switches a
# model's attention implementation to it. Requires CUDA and the sglang package (imported lazily).
"""Minimal patch: replace HF attention kernel with SGLang Triton extend_attention_fwd_unified.

Instead of monkey-patching the entire attention forward (qkv proj, norm, rope, o_proj),
we register a custom attention backend via HF's ALL_ATTENTION_FUNCTIONS. This way HF
keeps its own qkv_proj, norm, rope, o_proj logic; we ONLY replace the core attention
computation (Q@K^T softmax V) with extend_attention_fwd_unified.

The extend_attention_fwd_unified kernel uses per-request indptrs, which naturally
gives batch-invariant results (each request is computed independently).

The TritonAttnFunction autograd wrapper ensures:
  - Forward: uses SGLang Triton kernel (bitwise-aligned with inference)
  - Backward: computes correct gradients so qkv_proj receives gradient flow
"""

import torch


class TritonAttnFunction(torch.autograd.Function):
    # [CLASS DOC]
    # Custom autograd operation pairing the SGLang Triton forward with the project's tiled Triton
    # backward. Saves query, key, value and output for backward plus the batch and sequence length.
    # Inputs are token-flattened tensors (batch times sequence tokens, heads, head dim); the
    # integer arguments get no gradient.
    """Autograd wrapper: Triton forward + Triton backward.

    This gives us true on-policy (fwd matches inference exactly) while
    maintaining full gradient flow through q/k/v projections.
    The backward uses memory-efficient tiled Triton kernels (no SxS materialization).
    """

    @staticmethod
    def forward(ctx, q, k, v, B, S):
        # [FUNCTION DOC]
        # Builds the metadata the kernel expects for B contiguous equal-length sequences: query and
        # key/value index-pointer arrays at multiples of S (int32), a key/value index array over all
        # tokens (int64), and zero prefix lengths. Allocates an output like q, runs the unified
        # extend-attention kernel in causal mode with maximum extend length S (writes into the
        # output), saves tensors and sizes on the context and returns the output. The sglang import
        # is local (optional heavy dependency). Runs on the query's device. Assumes q, k, v are
        # contiguous bfloat16.
        """
        Args:
            q: [total, num_heads, D], bf16, contiguous
            k: [total, num_kv_heads, D], bf16, contiguous
            v: [total, num_kv_heads, D], bf16, contiguous
            B: batch size (int)
            S: sequence length (int)
        Returns:
            o: [total, num_heads, D], bf16
        """
        from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd_unified

        total = B * S
        device = q.device

        o = torch.empty_like(q)
        qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S
        kv_indptr = qo_indptr.clone()
        kv_indices = torch.arange(total, device=device, dtype=torch.int64)
        prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

        extend_attention_fwd_unified(
            q,
            o,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            max_len_extend=S,
            is_causal=True,
        )

        ctx.save_for_backward(q, k, v, o)
        ctx.B = B
        ctx.S = S
        return o

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Restores saved tensors and sizes and calls the project's dense attention backward kernel
        # with the output gradient to get gradients for q, k and v; returns them followed by two
        # none values for the batch and sequence length arguments. The kernel import is local.
        """Memory-efficient Triton backward for causal self-attention."""
        from miles.kernels.attention.dense.backward import dense_attention_backward

        q, k, v, o = ctx.saved_tensors
        B, S = ctx.B, ctx.S

        dq, dk, dv = dense_attention_backward(q, k, v, o, grad_output, B, S)
        return dq, dk, dv, None, None


def _sglang_triton_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # [FUNCTION DOC] HELPER: Attention function with the library's attention-interface signature.
    # Ignores the mask, scaling, dropout and extra keywords (deleted unused), so the kernel's own
    # default softmax scale applies and padding/mask information is NOT honored.
    # Reshapes query (batch, heads, seq, dim) and key/value (batch, kv heads, seq, dim) into
    # token-flattened (batch*seq, heads, dim) contiguous tensors, casts all to bfloat16 to match
    # the inference backend, applies the autograd function, and returns the output reshaped to
    # (batch, seq, heads, dim) paired with none for attention weights.
    # KNOWN ISSUE: the scaling argument is discarded, so a model whose scaling differs from the
    # kernel default (one over root of head dim) would silently get different results.
    """Drop-in replacement for HF's attention_interface.

    Input:  query [B, num_heads, S, D], key/value [B, num_kv_heads, S, D]
    Output: attn_output [B, S, num_heads, D], None
    """
    del attention_mask, scaling, dropout, kwargs

    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    total = B * S

    q = query.transpose(1, 2).contiguous().view(total, num_heads, D)
    k = key.transpose(1, 2).contiguous().view(total, num_kv_heads, D)
    v = value.transpose(1, 2).contiguous().view(total, num_kv_heads, D)

    # Force kernel inputs to bf16 right before extend_unified to match
    # the SGLang inference-side Triton backend behavior.
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    o = TritonAttnFunction.apply(q, k, v, B, S)

    attn_output = o.view(B, S, num_heads, D)
    return attn_output, None


def apply_sglang_triton_attention_patch(model):
    # [FUNCTION DOC]
    # Registers the function above under the name "triton" in the library's global attention
    # registry (import is local) and sets the model config's attention implementation to that name.
    # Returns the count of submodules having both query and output projections (an estimate of the
    # attention layers affected; nothing per-module is modified). Mutates global registry and config.
    """Register SGLang Triton as attention backend and activate it on the model."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["triton"] = _sglang_triton_attention
    model.config._attn_implementation = "triton"

    patched = sum(1 for _, m in model.named_modules() if hasattr(m, "q_proj") and hasattr(m, "o_proj"))
    return patched
