# [MODULE DOC]
# Quantizer for the compressed-tensors checkpoint format. For each converted weight whose base name
# is listed as a packed weight in the target checkpoint, replaces the weight with packed 4-bit
# data plus scale (and zero point / shape) tensors: either MXFP4 (via the project's MXFP4
# quantizer) or group-wise symmetric/asymmetric int4 simulated through a CUDA fake-quantization
# kernel and then packed eight values per 32-bit integer. Other tensors pass through. Also
# contains an AWQ-style 4-bit packed linear module and several int-packing helpers. The CUDA
# kernel module is an optional import (none if unavailable, failing only when int4 packing runs).
import logging
import math
import re

import torch
import torch.nn as nn

from miles.utils.mxfp4 import quantize_mxfp4

try:
    import fake_int4_quant_cuda
except ImportError:
    fake_int4_quant_cuda = None

logger = logging.getLogger(__name__)


__all__ = ["quantize_params_compressed_tensors"]


class WQLinear_GEMM(nn.Module):
    # [CLASS DOC] HELPER:
    # AWQ-style 4-bit weight-only linear layer container: registers integer buffers holding packed
    # quantized weights (in_features rows, out_features/8 packed columns, 32-bit), packed zero
    # points (groups by out_features/8) and half-precision scales (groups by out_features), plus an
    # optional half-precision bias. A group size of -1 means one group over all input features.
    # KNOWN ISSUE: not used by the quantizer function in this module (only constructed by its own
    # class method), so it appears to be vestigial.
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, training=False):
        # [FUNCTION DOC] Only 4-bit supported (raises not-implemented otherwise). Resolves group
        # size, asserts in_features divisible by it and out_features divisible by 8, and registers
        # zero-initialized buffers on the given device; bias buffer exists only if requested (else
        # the attribute is none). The training argument overwrites the module's training flag.
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training

        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        # [FUNCTION DOC]
        # Builds an instance sized like the given linear layer (on its weight's device). With
        # init_only returns it empty. Otherwise requires scales and zeros (assertion error), copies
        # scales (as half) and bias, then on the current CUDA device: expands scales (and tensor
        # zeros) per group along the input axis, quantizes the weight as round(weight / scale +
        # zero), transposes to (in, out), regroups output columns in blocks of eight and packs them
        # into 32-bit integers using the AWQ interleaved nibble order [0,4,1,5,2,6,3,7]; stores as
        # the weight buffer. Zero points are packed the same way when given as a tensor; when a
        # scalar, a packed constant value filled across groups is used. Returns the instance.
        # KNOWN ISSUE: the scalar-zero branch builds its fill from the scales' shape as if scales
        # were (groups, out) while the pack count assumes eight per word, and the nibble order map
        # has eight entries so only 4-bit works (enforced by the constructor).
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        repeat_scales = scales.to(device).t().repeat_interleave(group_size, 1)
        if isinstance(zeros, torch.Tensor):
            repeat_zeros = zeros.to(device).t().repeat_interleave(group_size, 1)
        else:
            repeat_zeros = zeros
        intweight = torch.round(linear.weight.to(device) / repeat_scales + repeat_zeros).to(torch.int).t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        del repeat_scales

        intweight = intweight.reshape(-1, intweight.shape[1] // pack_num, pack_num)

        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=device) * awq_linear.w_bit
        intweight = intweight << new_order_map
        intweight = torch.sum(intweight, dim=-1).to(torch.int32)
        awq_linear.qweight = intweight

        if isinstance(zeros, torch.Tensor):
            zeros = zeros.to(dtype=torch.int32, device=device)
            zeros = zeros.reshape(-1, zeros.shape[1] // pack_num, pack_num)
            zeros = zeros << new_order_map
            qzeros = torch.sum(zeros, dim=-1).to(torch.int32)

        else:
            value = 0
            for i in range(pack_num):
                value |= zeros << (i * awq_linear.w_bit)
            qzeros = (
                torch.ones(
                    (scales.shape[0], scales.shape[1] // pack_num),
                    dtype=torch.int32,
                    device=device,
                )
                * value
            )

        awq_linear.qzeros = qzeros

        return awq_linear


def pack_to_int32(
    value,
    num_bits,
    packed_dim=1,
    sym=False,
):
    # [FUNCTION DOC] HELPER:
    # Packs a matrix of small integers into 32-bit integers, 32 // num_bits values per word, lowest
    # index in the lowest bits. Raises value error if num_bits is above 8 or below 1. If sym is set,
    # first adds 2^(num_bits-1) to shift signed values to unsigned (as unsigned 8-bit). If the packed
    # dimension is 0, transposes first (and back at the end). Pads the packed axis with zeros up to
    # a multiple of the pack factor, views as (rows, words, pack factor), casts to 32-bit signed,
    # shifts each position by index times num_bits and sums over the pack axis. Returns the packed
    # int32 matrix (rows, words), transposed back for dim 0.
    # if value.dtype is not torch.int8:
    #     raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    if sym:
        offset = 1 << (num_bits - 1)
        value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def round_to_quantized_type_dtype(
    tensor,
    dtype,
    cast_to_original_dtype=False,
):
    # [FUNCTION DOC] HELPER: Clamps the tensor to the integer range of the target integer dtype,
    # rounds to nearest, and casts to that dtype; optionally casts back to the input dtype.
    original_dtype = tensor.dtype
    iinfo = torch.iinfo(dtype)
    rounded = torch.round(torch.clamp(tensor, iinfo.min, iinfo.max)).to(dtype)
    if cast_to_original_dtype:
        return rounded.to(original_dtype)
    return rounded


@torch.no_grad()
def quantize(
    x,
    scale,
    zero_point,
    dtype=torch.int8,
):
    # [FUNCTION DOC] HELPER (no gradient tracking): Group-wise quantization of the last axis. Group
    # size = last size of x divided by last size of the scale (integer division). Splits the last
    # axis into (groups, group size), divides by the scale (broadcast over the group axis), adds the
    # zero point (cast to x dtype, in place on the scaled copy) when given, then rounds and clamps
    # to the integer dtype range and flattens the last two axes back. Returns a tensor of the
    # target dtype (default signed 8-bit) with the same shape as x.
    # KNOWN ISSUE: the initial zero-filled output allocation is dead code (immediately replaced).
    group_size = x.shape[-1] // scale.shape[-1]
    output_dtype = dtype
    output = torch.zeros_like(x).to(output_dtype)

    reshaped_dims = (
        math.ceil(x.shape[-1] / group_size),
        group_size,
    )
    x = x.unflatten(-1, reshaped_dims)

    scaled = x / scale.unsqueeze(-1)

    if zero_point is not None:
        zero_point = zero_point.unsqueeze(-1)
        scaled += zero_point.to(x.dtype)

    # clamp and round
    output = round_to_quantized_type_dtype(tensor=scaled, dtype=dtype)

    output = output.flatten(start_dim=-2)
    output = output.to(output_dtype)

    return output


def if_quant(name, patterns):
    # [FUNCTION DOC] HELPER: True if any regular expression in the list is found anywhere in the
    # name. KNOWN ISSUE: not used in this module.
    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


def pack_layer(weight, group_size, sym=True):
    # [FUNCTION DOC] HELPER:
    # Input: 2-D weight (rows, cols; cols divisible by group size), group size, symmetric flag.
    # Calls the CUDA fake-int4 kernel (one group of rows by group-size columns) to get fake-
    # quantized values, scales and zero points; reshapes each to (rows, 1, groups, ...); dequantizes
    # back to weight scale (value times scale if symmetric, else (value minus zero point) times
    # scale) and reshapes to the weight shape; scale becomes (rows, groups) contiguous. For
    # asymmetric: zero points are transposed, cast to int32 and packed eight per word with the AWQ
    # nibble order [0,4,1,5,2,6,3,7] (shifts of 4 bits); for symmetric no zero point. Then
    # re-quantizes the dequantized weight with the scale (and zero point) into signed 8-bit
    # (symmetric) or unsigned 8-bit (asymmetric) integers and packs them into 32-bit words with 4
    # bits each (the symmetric case offset by 8). Returns (packed weight, scale, packed zero point
    # or none).
    # KNOWN ISSUE: raises an attribute error if the optional CUDA kernel module failed to import
    # (it is then none); asymmetric zero-point packing assumes the group count is divisible by 8,
    # and the zero-point passed to the re-quantization step is the unpacked float-valued one.
    w, scale, zp = fake_int4_quant_cuda.fake_int4_quant_cuda(weight, (1, group_size), sym)
    w = w.view(weight.shape[0], 1, weight.shape[1] // group_size, group_size)
    scale = scale.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    zp = zp.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    if sym:
        w = w * scale
    else:
        w = (w - zp) * scale
    w = w.view(weight.shape)
    scale = scale.view(weight.shape[0], -1).contiguous()
    if not sym:
        zp = zp.view(weight.shape[0], -1)
        zeros = zp.t().contiguous().to(torch.float32)
        zeros = zeros.to(dtype=torch.int32, device=w.device)
        zeros = zeros.reshape(-1, zeros.shape[1] // 8, 8)
        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=zeros.device) * 4
        zeros = zeros << new_order_map
        packed_zp = torch.sum(zeros, dim=-1).to(torch.int32)
    else:
        zp = None
        packed_zp = None

    quantized_weight = quantize(
        x=w,
        scale=scale,
        zero_point=zp,
        dtype=torch.int8 if sym else torch.uint8,
    )
    packed_weight = pack_to_int32(quantized_weight, 4, sym=sym)
    return packed_weight, scale, packed_zp


def quantize_params_compressed_tensors(converted_named_params, quantization_config, packed_weight_basenames):
    # [FUNCTION DOC]
    # Reads the format, group size and symmetric flag from the config's first weight group. For the
    # "mxfp4-pack-quantized" format asserts float type, 4 bits, uint8 scale dtype and symmetric. For
    # each (name, tensor): if the name ends with ".weight" and its stem is in the packed base-name
    # collection, replaces it: MXFP4 gives "<stem>.weight_packed" and "<stem>.weight_scale"; int4
    # gives (optionally) "<stem>.weight_zero_point" when asymmetric, then ".weight_packed",
    # ".weight_scale" and ".weight_shape" (original shape as an int32 tensor on the CUDA device);
    # all other pairs pass through unchanged. Returns the new list in order.
    quant_format = quantization_config["format"]
    w_cfg = quantization_config["config_groups"]["group_0"]["weights"]
    group_size = w_cfg["group_size"]
    is_symmetric = w_cfg["symmetric"]
    is_mxfp4 = quant_format == "mxfp4-pack-quantized"
    if is_mxfp4:
        assert w_cfg["type"] == "float"
        assert w_cfg["num_bits"] == 4
        assert w_cfg["scale_dtype"] == "torch.uint8"
        assert is_symmetric
    results = []

    for name, param in converted_named_params:
        if not (name.endswith(".weight") and name.removesuffix(".weight") in packed_weight_basenames):
            results.append((name, param))
            continue

        qweight_name = name.replace(".weight", ".weight_packed")
        scale_name = name.replace(".weight", ".weight_scale")
        if is_mxfp4:
            qw, s = quantize_mxfp4(param, group_size)
            results.append((qweight_name, qw))
            results.append((scale_name, s))
        else:
            qw, s, zp = pack_layer(param, group_size, is_symmetric)
            weight_shape = torch.tensor(param.shape, dtype=torch.int32, device="cuda")
            weight_shape_name = name.replace(".weight", ".weight_shape")
            if zp is not None:
                zp_name = name.replace(".weight", ".weight_zero_point")
                results.append((zp_name, zp))
            results.append((qweight_name, qw))
            results.append((scale_name, s))
            results.append((weight_shape_name, weight_shape))

    return results
