# [MODULE DOC] HELPER: Linear projection with bfloat16 inputs and float32 output, matching the
# numerics of the SGLang rollout engine's compressor so training log-probs align with inference.
# Provides a custom autograd function (bf16 forward, float32 backward) and a functional wrapper.
import torch


class _BFloat16LinearFP32Func(torch.autograd.Function):
    # [CLASS DOC] HELPER: Custom autograd function computing activation times transposed weight, no
    # bias, for arbitrary leading dimensions. Forward casts both operands to bf16 and produces float32.
    # Backward upcasts everything to float32 and returns gradients cast to the original dtypes of the
    # input and weight.
    # Forward matches SGLang's default DeepSeek-V4 compressor path
    # (`sglang.jit_kernel.deepseek_v4.linear_bf16_fp32`, cublas backend):
    # BF16 activation x BF16 weight -> FP32 output. This keeps Megatron's
    # compressor log-prob computation aligned with SGLang rollout. Backward is
    # only needed for training, so keep its gradient matmuls in FP32.
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Casts the input and the weight (out features by in features) to bf16,
        # saving the bf16 copies plus the original input shape and the original dtypes for backward.
        # Flattens leading dims to a 2D matrix and multiplies by the transposed weight, with float32
        # output. On ROCm (no bf16-in float32-out matmul) the bf16-rounded values are upcast to float32
        # before the matmul. Returns float32 with the input's leading dims and last dim equal to out
        # features.
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        ctx.save_for_backward(x_bf16, weight_bf16)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype

        x_2d = x_bf16.reshape(-1, x_bf16.shape[-1])
        if torch.version.hip is not None:
            # ROCm lacks bf16-in/fp32-out torch.mm, so upcast bf16-rounded inputs to fp32 to match CUDA/cublas fp32 accumulation.
            out = torch.mm(x_2d.float(), weight_bf16.t().float())
        else:
            out = torch.mm(x_2d, weight_bf16.t(), out_dtype=torch.float32)
        return out.view(*x.shape[:-1], weight_bf16.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # [FUNCTION DOC] HELPER: Flattens the output gradient to 2D float32. If the input needs a
        # gradient: multiply by the saved bf16 weight (as float32), restore the input shape, and cast
        # to the input's original dtype. If the weight needs a gradient: transposed output gradient
        # times the saved bf16 input (as float32), cast to the weight's original dtype. Returns the
        # pair (input gradient, weight gradient), with nothing for any not required.
        x_bf16, weight_bf16 = ctx.saved_tensors
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_2d.matmul(weight_bf16.float())
            grad_x = grad_x.view(ctx.input_shape).to(ctx.input_dtype)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            x_2d = x_bf16.reshape(-1, x_bf16.shape[-1])
            grad_weight = grad_output_2d.t().matmul(x_2d.float()).to(ctx.weight_dtype)

        return grad_x, grad_weight


def linear_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Functional entry point; applies the custom autograd function to the
    # activation and weight and returns the float32 result.
    return _BFloat16LinearFP32Func.apply(x, weight)
