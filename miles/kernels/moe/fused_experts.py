# [MODULE DOC] Differentiable mixture-of-experts feed-forward built from sglang's fused Triton MoE
# forward kernels plus this package's own backward kernels (fused_experts_bwd). The computation is
# split into four autograd functions so each stage has an explicit backward: (1) gate/up projection
# of every (token, selected expert) pair with the expert's first weight; (2) SiLU-and-multiply
# (gate half times up half); (3) down projection with the second weight, multiplied by the routing
# weight; (4) sum over the selected experts. The public function fused_experts chains them.
# Conventions: hidden states are tokens by hidden size (bfloat16, contiguous); first expert weights
# are experts by (2 times intermediate size) by hidden size, with gate rows first then up rows;
# second expert weights are experts by hidden size by intermediate size; routing weights and expert
# ids are tokens by top-k. Forward and backward process tokens in chunks of 65536 and use a fixed
# tile configuration (64 by 64 by 32, group size 8) chosen for determinism, not speed.
from __future__ import annotations

import torch
import triton.language as tl
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce,
    silu_and_mul,
)

from miles.kernels.moe.fused_experts_bwd import invoke_fused_moe_backward_kernel


class GateUpProjFunction(torch.autograd.Function):
    # [CLASS DOC] Stage 1 autograd function: for each token and each of its selected experts, the
    # product of the token's hidden state with the expert's first weight matrix, giving one row of
    # width N (gate and up halves) per (token, expert) pair. Routing weights are not applied here and
    # receive an all-zero gradient from this stage. Saves the inputs and the tile configuration.
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        # [FUNCTION DOC] Returns a new tensor of shape (tokens times top-k, N) in the hidden-state
        # dtype; row (token times top-k plus slot) holds the product with the expert chosen in that
        # slot. For each 65536-token chunk it builds the block-aligned sorted token/expert metadata
        # and runs sglang's fused MoE kernel (bfloat16 compute, no quantization, routed weight NOT
        # multiplied, top-k as the slot count). All pairs are processed, none dropped.
        # KNOWN ISSUE: the chunk loop runs one extra iteration, so when the token count is an exact
        # multiple of the chunk size (including zero tokens) the last chunk is empty and still gets
        # metadata built and a kernel launch on zero rows; the backward loops skip empty chunks.
        num_tokens, _ = hidden_states.shape
        E, N, _ = w1.shape
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        topk = topk_ids.shape[1]

        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            cur_intermediate_cache1 = intermediate_cache1[begin_chunk_idx * topk : end_chunk_idx * topk]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            invoke_fused_moe_kernel(
                curr_hidden_states,
                w1,
                None,
                cur_intermediate_cache1,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                topk_ids.shape[1],
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Chunked backward with this package's Triton kernel. Allocates zero gradients
        # for hidden states, the first weights and the routing weights (the last stays zero). For each
        # non-empty chunk it rebuilds the aligned metadata, allocates a zero chunk gradient for the
        # hidden states and a zero full-size gradient for the weights, calls the backward kernel
        # without routed-weight multiplication and without a routing-weight gradient, then adds the
        # chunk results into the totals (hidden-state gradient into its slice, weight gradient summed).
        # Returns (hidden-state gradient, weight gradient, zero routing-weight gradient, none).
        # The per-chunk weight-gradient buffer is the same size as the whole expert weight tensor, so
        # peak memory scales with it.
        """
        Backward pass for GateUpProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens * topk, N)

        Returns:
            (grad_hidden_states, grad_w1, grad_topk_weights, None)
        """

        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, N, D_in = w1.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_w1 = torch.zeros_like(w1)
        # GateUpProj stage doesn't need topk_weights gradient
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            # Get aligned metadata
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # Prepare gradient buffer for this chunk
            curr_grad_hidden_states = torch.zeros_like(curr_hidden_states)
            curr_grad_w1 = torch.zeros_like(w1)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=False
            # Use chunk of hidden_states to match sorted_token_ids indices
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_hidden_states,  # Use chunk of hidden_states to match sorted_token_ids
                weight=w1,
                grad_input=curr_grad_hidden_states,
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,  # Not needed for GateUpProj
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate gradients
            grad_hidden_states[begin_chunk_idx:end_chunk_idx] += curr_grad_hidden_states
            grad_w1 += curr_grad_w1

        return grad_hidden_states, grad_w1, grad_topk_weights, None


class SiluAndMulFunction(torch.autograd.Function):
    # [CLASS DOC] Stage 2 autograd function: gated activation. For rows of width N, output width N/2 is
    # SiLU(first half, the gate) times second half (the up part). Saves only its input.
    @staticmethod
    def forward(ctx, intermediate_cache1: torch.Tensor):
        # [FUNCTION DOC] Allocates the half-width output in the input dtype and fills it with sglang's
        # fused SiLU-and-multiply kernel over the input viewed as rows of width N (so the input must
        # be viewable, i.e. contiguous). Returns the output.
        num_tokens, N = intermediate_cache1.shape
        intermediate_cache2 = torch.empty(
            (num_tokens, N // 2),
            device=intermediate_cache1.device,
            dtype=intermediate_cache1.dtype,
        )
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        ctx.save_for_backward(intermediate_cache1)
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Plain torch backward. With x1 the gate half and x2 the up half: gradient of
        # the gate half = incoming gradient times x2 times the SiLU derivative (sigmoid(x1) plus x1
        # times sigmoid times (1 minus sigmoid)); gradient of the up half = incoming gradient times
        # SiLU(x1). The two are concatenated along the last dimension and shaped like the saved input.
        (intermediate_cache1,) = ctx.saved_tensors
        N = intermediate_cache1.shape[-1]
        x1, x2 = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        silu_x1 = torch.nn.functional.silu(x1)

        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        grad_x1 = grad_output * x2 * dsilu_dx1
        grad_x2 = grad_output * silu_x1
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)

        return grad_input.view_as(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    # [CLASS DOC] Stage 3 autograd function: projects every (token, expert) row of the activated
    # intermediate back to the hidden size with the expert's second weight matrix and multiplies it
    # by that pair's routing weight. Output is a per-token, per-slot tensor (not yet summed over
    # experts). Saves its inputs and the tile configuration.
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        # [FUNCTION DOC] The token count is the input row count divided by top-k. Returns a new
        # tensor of shape (tokens, top-k, hidden size) in the input dtype. Per 65536-token chunk,
        # builds aligned metadata and runs sglang's fused MoE kernel with routed-weight
        # multiplication on and a slot count of 1 (rows are already one per pair), bfloat16 compute,
        # no quantization and no tensor-memory-accelerator loads.
        # KNOWN ISSUE: as in the first stage, the chunk loop has one extra iteration, so an exact
        # multiple of the chunk size (including zero tokens) yields an empty last chunk that is not
        # skipped here, whereas the backward skips empty chunks.
        num_tokens, _ = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens //= topk
        E, _, _ = w2.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            cur_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            cur_intermediate_cache3 = intermediate_cache3[begin_chunk_idx:end_chunk_idx]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )
            invoke_fused_moe_kernel(
                cur_intermediate_cache2,
                w2,
                None,
                cur_intermediate_cache3,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                a_use_tma=False,
                b_use_tma=False,
            )

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Chunked backward with this package's Triton kernel. Allocates zero gradients
        # for the input rows, the second weights and the routing weights. For each non-empty chunk:
        # rebuilds aligned metadata, allocates zero chunk buffers (input-row gradient, a full-size
        # weight gradient, routing-weight gradient), runs the backward kernel with routed-weight
        # multiplication on and a slot count of 1, then writes the chunk's input-row and routing-weight
        # gradients into their slices and adds the chunk's weight gradient into the total.
        # Returns (input-row gradient, weight gradient, routing-weight gradient, none).
        """
        Backward pass for DownProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens, topk, hidden_size)

        Returns:
            (grad_intermediate_cache2, grad_w2, grad_topk_weights, None)
        """
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, hidden_size, intermediate_size = w2.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        grad_w2 = torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]

            # Get aligned metadata
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # Prepare gradient buffers for this chunk
            curr_grad_intermediate_cache2 = torch.zeros_like(curr_intermediate_cache2)
            curr_grad_w2 = torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=True
            # Note: Use top_k=1 to match forward pass indexing
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_intermediate_cache2,
                weight=w2,
                grad_input=curr_grad_intermediate_cache2,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate gradients
            grad_intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk] = curr_grad_intermediate_cache2
            grad_w2 += curr_grad_w2
            grad_topk_weights[begin_chunk_idx:end_chunk_idx] = curr_grad_topk_weights

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


class MoeSumReduceFunction(torch.autograd.Function):
    # [CLASS DOC] Stage 4 autograd function: sums the per-slot expert outputs of each token.
    @staticmethod
    def forward(
        ctx,
        intermediate_cache3: torch.Tensor,
        hidden_states_shape,
    ):
        # [FUNCTION DOC] Allocates an output of the given shape (tokens by hidden size) in the input
        # dtype and fills it with sglang's sum-reduce kernel over the top-k axis with a scaling factor
        # of 1.0. Saves the input only for its shape in backward. Returns the output.
        out_hidden_states = torch.empty(
            hidden_states_shape, device=intermediate_cache3.device, dtype=intermediate_cache3.dtype
        )
        moe_sum_reduce(
            intermediate_cache3,
            out_hidden_states,
            1.0,
        )
        ctx.save_for_backward(intermediate_cache3)
        return out_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] The gradient of a sum is the incoming gradient broadcast to every slot: returns
        # it as a non-contiguous expanded view with the saved input's shape (tokens, top-k, hidden
        # size), and none for the shape argument.
        (intermediate_cache3,) = ctx.saved_tensors
        return grad_output.unsqueeze(1).expand_as(intermediate_cache3), None


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    # [FUNCTION DOC] Public entry: full expert feed-forward. Inputs: hidden_states (tokens by hidden,
    # bfloat16), w1 (experts by 2 times intermediate by hidden), w2 (experts by hidden by
    # intermediate), topk_weights and topk_ids (tokens by top-k routing weights and expert ids).
    # Raises assertion errors if the hidden sizes differ, the routing tensors differ in shape, any of
    # hidden_states, w1, w2 is not contiguous, or hidden_states is not bfloat16 (the shape of w2
    # relative to w1 is not checked). Chains gate/up projection, SiLU-and-multiply, down projection
    # with routing weights, and the sum over experts. Returns tokens by hidden; differentiable in
    # hidden_states, w1, w2 and topk_weights (routing weights get gradient from the down projection).
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.bfloat16]

    intermediate_cache1 = GateUpProjFunction.apply(
        hidden_states,
        w1,
        topk_weights,
        topk_ids,
    )
    intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
    intermediate_cache3 = DownProjFunction.apply(
        intermediate_cache2,
        w2,
        topk_weights,
        topk_ids,
    )
    output_hidden_states = MoeSumReduceFunction.apply(
        intermediate_cache3,
        hidden_states.shape,
    )
    return output_hidden_states
