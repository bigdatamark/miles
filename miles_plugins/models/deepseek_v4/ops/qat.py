# [MODULE DOC] HELPER: Fake-quantization (quantization-aware-training simulation) for DeepSeek-V4
# activations: quantizes to 8-bit float and immediately dequantizes, so downstream computation sees
# fp8 rounding error while staying in the original dtype. Exposes the simulation function and a
# straight-through autograd wrapper (gradient passes unchanged). Depends on the external tile kernels
# package and the in-repo activation quantization kernel (GPU).
import torch

from tile_kernels.quant import per_token_cast_back

from miles.kernels.quant.fp8_act_quant import act_quant


def fp8_simulate(x: torch.Tensor, block_size: int):
    # [FUNCTION DOC] HELPER: Makes the input contiguous, quantizes it per block of block_size along the
    # last dim to e4m3 with power-of-two (ue8m0) scales via the in-repo kernel, flattens to a 2D matrix
    # with one row per token plus a scale matrix with last-dim/block_size columns, then dequantizes
    # with the external cast-back kernel to bf16 if the input is bf16, otherwise float32. Returns a
    # tensor shaped like the contiguous input and cast to the input dtype. The last dim must be
    # divisible by the block size. GPU kernels are launched; the input is not modified.
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    Both the cast (via :func:`act_quant`) and the cast-back step are routed
    through ``deepseek-ai/TileKernels`` so we share the same FP8 kernels with
    the rest of the DeepSeek stack.
    """
    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, "ue8m0")

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    out_flat = per_token_cast_back((y_flat, scale_flat), "bf16" if x.dtype == torch.bfloat16 else "fp32", block_size)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    # [CLASS DOC] HELPER: Straight-through estimator: forward applies the fp8 simulation, backward
    # returns the incoming gradient unchanged for the tensor and nothing for the block size.
    @staticmethod
    def forward(ctx, kv, block_size=128):
        # [FUNCTION DOC] HELPER: Returns the fp8-simulated copy of kv using the given block size
        # (default 128). Nothing is saved for backward.
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        # [FUNCTION DOC] HELPER: Passes the gradient through unchanged; no gradient for block size.
        return grad_kv, None


# module-level alias of the function entry point (takes the tensor and optionally block size)
fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
