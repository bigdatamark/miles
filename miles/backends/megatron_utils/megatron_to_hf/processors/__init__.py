# [PACKAGE DOC]
# Post-processing applied to converted Hugging Face tensors: vocabulary padding removal, and
# quantizers for several target formats (block/per-tensor FP8, MXFP8, NVFP4, and compressed-tensors
# int4/MXFP4 packing). Each quantizer takes (args, Megatron parameter name, list of (HF name,
# tensor) pairs, quantization config) and returns a possibly longer list in which quantized weights
# are accompanied by their scale tensors; the dispatcher defined here selects the quantizer by the
# config's method.
from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import quantize_params_fp8
from .quantizer_mxfp8 import quantize_params_mxfp8
from .quantizer_nvfp4 import quantize_params_nvfp4

__all__ = [
    "remove_padding",
    "quantize_param",
    "quantize_params_fp8",
    "quantize_params_mxfp8",
    "quantize_params_nvfp4",
    "quantize_params_compressed_tensors",
]


def quantize_params(args, megatron_name, converted_named_params, quantization_config, packed_weight_basenames=None):
    # [FUNCTION DOC]
    # Dispatcher. With no quantization config returns the input list unchanged. Otherwise selects by
    # the config's "quant_method" entry, tested in this order: "fp8", "mxfp8", NVFP4 (either the
    # config's "quant_algo" equals "NVFP4" or the method is "nvfp4"), "compressed-tensors" (asserts
    # the packed weight base names were provided, then quantizes without args or the Megatron name).
    # Returns the chosen quantizer's list of (name, tensor) pairs.
    # KNOWN ISSUE: the method entry is read by key access in the first tests, so a config that has
    # only "quant_algo" (no "quant_method") raises a key error before the NVFP4 test is reached; an
    # unrecognized method falls off the end and returns none instead of raising. Also the package's
    # public-names list contains "quantize_param" (no such name) and omits this function.
    if quantization_config is None:
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "mxfp8":
        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config.get("quant_algo") == "NVFP4" or quantization_config["quant_method"] == "nvfp4":
        return quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        assert (
            packed_weight_basenames is not None
        ), "compressed-tensors quantization needs the checkpoint's packed weight names"
        return quantize_params_compressed_tensors(converted_named_params, quantization_config, packed_weight_basenames)
