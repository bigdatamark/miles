# [MODULE DOC] Command-line tool converting a Hugging Face safetensors checkpoint to MXFP8
# (fp8 values with per-32-element power-of-two scales stored as unsigned bytes). Sources:
# BF16/FP16/FP32 checkpoints, or block-scaled FP8 with ue8m0 scales and 128 by 128 blocks (scales
# reused by expansion when they are power-of-two, otherwise dequantized and requantized).
# Excluded from quantization: names with norm, embed, router, head and similar substrings,
# optional extra substrings, and optionally the first/last N decoder layers (kept in BF16; FP8
# sources are dequantized to BF16 for them). Needs CUDA. Writes updated config with a
# quantization section and a new weight index (total size in bytes).
"""
python tools/convert_hf_to_mxfp8.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR]
                                    [--extra-high-precision-layers-hf ...]

Convert an HF safetensors checkpoint to MXFP8 with UE8M0 scales.
Supported input checkpoint types:
  1) BF16/FP16/FP32 checkpoints
  2) Block-scaled FP8 checkpoints with UE8M0 scales and block size [128, 128]
"""

import argparse
import gc
import json
import os
import re
import shutil


import safetensors
import safetensors.torch
import torch
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from tqdm import tqdm

from miles.utils.mxfp8 import MXFP8_GROUP_SIZE
from miles.utils.mxfp8 import mxfp8_quantize as quantize_mxfp8


SKIP_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
    "head.",
    "wo_a",
    "ffn.gate.",
    "compressor.",
)

SOURCE_FP8_BLOCK_SIZE = [128, 128]
TARGET_MXFP8_BLOCK_SIZE = [1, MXFP8_GROUP_SIZE]
SOURCE_FP8_SCALE_KEY_SUFFIX = ".weight_scale_inv"
SOURCE_FP8_DTYPES = (torch.float8_e4m3fn,) + ((torch.float8_e4m3fnuz,) if hasattr(torch, "float8_e4m3fnuz") else ())


def _strip_weight_suffix(weight_key: str) -> str:
    # [FUNCTION DOC] HELPER: Removes the trailing ".weight"; raises a value error if absent.
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected key ending with '.weight', got: {weight_key}")
    return weight_key[: -len(".weight")]


def _is_source_block_fp8_ue8m0_checkpoint(cfg: dict) -> bool:
    # [FUNCTION DOC] HELPER: True when the config's quantization section has method fp8, block
    # size exactly 128 by 128, and scale format ue8m0.
    qcfg = cfg.get("quantization_config", {}) if isinstance(cfg, dict) else {}
    return (
        qcfg.get("quant_method") == "fp8"
        and list(qcfg.get("weight_block_size", [])) == SOURCE_FP8_BLOCK_SIZE
        and qcfg.get("scale_fmt") == "ue8m0"
    )


def _is_bf16_source_checkpoint(cfg: dict) -> bool:
    # [FUNCTION DOC] HELPER: True when the config has no (or an empty/non-mapping) quantization
    # section, or its method is none, empty or "bf16".
    qcfg = cfg.get("quantization_config", {}) if isinstance(cfg, dict) else {}
    if not isinstance(qcfg, dict) or not qcfg:
        return True
    return qcfg.get("quant_method") in (None, "", "bf16")


def _load_source_scale_u8(
    weights: dict[str, torch.Tensor],
    weight_key: str,
    weight: torch.Tensor,
    source_scale_index: dict[str, str],
    input_path: str,
    device: str,
    current_filename: str,
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    # [FUNCTION DOC] HELPER: Fetches the block scale of a source FP8 weight (name with ".weight"
    # replaced by ".weight_scale_inv"), from the current file's loaded tensors or by opening the
    # file the index says holds it. Crops to the number of 128-blocks of the weight. If the
    # scale is stored as unsigned bytes (exponents), returns (float scale = exponent shifted into
    # the float exponent field, the bytes, scale key). If float32, returns (the float scale,
    # the exponent bytes if every value is non-negative with zero mantissa, i.e. an exact power
    # of two, else none, key); other dtypes fail an assertion.
    scale_key = _strip_weight_suffix(weight_key) + SOURCE_FP8_SCALE_KEY_SUFFIX
    scale_file = source_scale_index[scale_key]
    if scale_file == current_filename and scale_key in weights:
        scale = weights[scale_key]
    else:
        with safetensors.safe_open(os.path.join(input_path, scale_file), framework="pt", device=device) as f:
            scale = f.get_tensor(scale_key)

    if scale.dtype == torch.uint8:
        scale_u8 = scale
    else:
        assert scale.dtype == torch.float32
        scale_u8 = None
        n, k = weight.shape[-2], weight.shape[-1]
        n_tiles = (n + SOURCE_FP8_BLOCK_SIZE[0] - 1) // SOURCE_FP8_BLOCK_SIZE[0]
        k_tiles = (k + SOURCE_FP8_BLOCK_SIZE[1] - 1) // SOURCE_FP8_BLOCK_SIZE[1]
        scale_fp32 = scale[..., :n_tiles, :k_tiles].contiguous()
        bits = scale_fp32.contiguous().view(torch.int32)
        mantissa_all_zero = not torch.any((bits & 0x007FFFFF) != 0).item()
        non_negative = not torch.any(bits < 0).item()
        if mantissa_all_zero and non_negative:
            scale_u8 = ((bits >> 23) & 0xFF).to(torch.uint8)
        return scale_fp32, scale_u8, scale_key

    n, k = weight.shape[-2], weight.shape[-1]
    n_tiles = (n + SOURCE_FP8_BLOCK_SIZE[0] - 1) // SOURCE_FP8_BLOCK_SIZE[0]
    k_tiles = (k + SOURCE_FP8_BLOCK_SIZE[1] - 1) // SOURCE_FP8_BLOCK_SIZE[1]
    scale_u8 = scale_u8[..., :n_tiles, :k_tiles].contiguous()
    scale_fp32 = (scale_u8.to(torch.int32) << 23).view(torch.float32)
    return scale_fp32, scale_u8, scale_key


def _source_fp8_to_mxfp8_scale_u8(weight: torch.Tensor, source_scale_u8: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Expands 128 by 128 block exponent scales to per-row, per-32-column
    # scales by repeating 128 times along rows and 4 times along columns, then crops to the
    # weight's rows and columns divided by 32. Returns a contiguous byte tensor.
    n, k = weight.shape[-2], weight.shape[-1]
    mxfp8_scale_u8 = source_scale_u8.repeat_interleave(SOURCE_FP8_BLOCK_SIZE[0], dim=-2).repeat_interleave(
        SOURCE_FP8_BLOCK_SIZE[1] // TARGET_MXFP8_BLOCK_SIZE[1], dim=-1
    )
    return mxfp8_scale_u8[..., :n, : (k // TARGET_MXFP8_BLOCK_SIZE[1])].contiguous()


def should_quantize(
    name: str,
    weight: torch.Tensor,
    skip_weight_substrings=SKIP_WEIGHT_SUBSTRINGS,
    allow_source_fp8: bool = False,
) -> bool:
    # [FUNCTION DOC] Decides whether a tensor is converted: true only if the name ends with
    # ".weight", contains none of the skip substrings, the dtype is 16/32-bit float (or FP8 when
    # allowed), it has at least two dims, and the last dim is divisible by 32.
    allowed_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if allow_source_fp8:
        allowed_dtypes += SOURCE_FP8_DTYPES
    if not name.endswith(".weight"):
        return False
    if any(substr in name for substr in skip_weight_substrings):
        return False
    if weight.dtype not in allowed_dtypes:
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % 32 != 0:
        return False
    return True


class ConversionResult:
    # [CLASS DOC] HELPER: Accumulates the tensor-to-file map, total byte size and list of module
    # names left unconverted. Not thread-safe (files are processed sequentially).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty map, zero size and empty list.
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(
        self,
        filename: str,
        q_weights: dict[str, torch.Tensor],
        module_names: list[str],
    ) -> None:
        # [FUNCTION DOC] HELPER: Records each tensor's file, adds its byte size, and extends the
        # unconverted module list.
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def process_file(
    input_path: str,
    output_path: str,
    filename: str,
    result_collector: ConversionResult,
    device: str,
    num_hidden_layers: int,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
    source_is_block_fp8_ue8m0: bool,
    extra_high_precision_layers_hf: tuple[str, ...],
    source_scale_index: dict[str, str],
) -> None:
    # [FUNCTION DOC] Converts one safetensors file. Loads all tensors on the device. Computes the
    # layer-name prefixes of the first and last N layers to keep in BF16 (recorded as unconverted
    # modules) and combines them with the fixed and extra skip substrings. For each weight
    # tensor: if convertible, an FP8 source with a power-of-two scale keeps its FP8 data and gets
    # expanded exponent scales; an FP8 source with other scales is dequantized to float32 and
    # requantized; a high-precision source is quantized directly; the scale is stored under the
    # ".weight_scale_inv" name. If not convertible, the module name (minus ".weight", skipping
    # expert weights) is recorded as unconverted and the tensor is kept, or dequantized to BF16
    # if it was source FP8. Non-weight tensors are copied, except source block-scale tensors of
    # FP8 sources which are dropped. Saves the file and reports to the collector.
    # KNOWN ISSUE: in the FP8-source path the scale tensor may be stored under the original
    # scale key found by lookup, but the scale for the non-FP8 dtype source path uses the
    # same suffix; behavior is consistent only because both use the same key name.
    weights: dict[str, torch.Tensor] = {}
    q_weights: dict[str, torch.Tensor] = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    modules_to_not_convert: list[str] = []
    head_end_idx = num_layers_at_start_in_bf16
    tail_start_idx = num_hidden_layers - num_layers_at_end_in_bf16
    dynamic_skip_layer_prefixes: set[str] = set()
    dynamic_skip_layer_prefixes.update({f"model.layers.{i}." for i in range(0, head_end_idx)})
    dynamic_skip_layer_prefixes.update({f"model.layers.{i}." for i in range(tail_start_idx, num_hidden_layers)})

    if num_layers_at_end_in_bf16 > 0 or num_layers_at_start_in_bf16 > 0:
        modules_to_not_convert.extend(sorted(dynamic_skip_layer_prefixes))

    dynamic_skip_substrings = (
        *SKIP_WEIGHT_SUBSTRINGS,
        *extra_high_precision_layers_hf,
        *sorted(dynamic_skip_layer_prefixes),
    )

    for key, tensor in weights.items():
        if not key.endswith(".weight"):
            continue

        should_quant = should_quantize(
            key,
            tensor,
            skip_weight_substrings=dynamic_skip_substrings,
            allow_source_fp8=source_is_block_fp8_ue8m0,
        )

        if should_quant:
            if source_is_block_fp8_ue8m0 and tensor.dtype in SOURCE_FP8_DTYPES:
                source_scale_fp32, source_scale_u8, scale_key = _load_source_scale_u8(
                    weights,
                    key,
                    tensor,
                    source_scale_index=source_scale_index,
                    input_path=input_path,
                    device=device,
                    current_filename=filename,
                )
                if source_scale_u8 is not None:
                    qweight = tensor.contiguous()
                    scale = _source_fp8_to_mxfp8_scale_u8(tensor, source_scale_u8)
                else:
                    # dequant to fp32, requant to mxfp8
                    weight_fp32 = block_quant_dequant(
                        tensor, source_scale_fp32, SOURCE_FP8_BLOCK_SIZE, torch.float32
                    ).contiguous()
                    qweight, scale = quantize_mxfp8(weight_fp32)
                q_weights[key] = qweight
                q_weights[scale_key] = scale
            else:
                qweight, scale = quantize_mxfp8(tensor)
                q_weights[key] = qweight
                q_weights[_strip_weight_suffix(key) + SOURCE_FP8_SCALE_KEY_SUFFIX] = scale
        else:
            if ".experts." not in key:
                modules_to_not_convert.append(_strip_weight_suffix(key))
            if source_is_block_fp8_ue8m0 and tensor.dtype in SOURCE_FP8_DTYPES:
                source_scale_fp32, _, _ = _load_source_scale_u8(
                    weights,
                    key,
                    tensor,
                    source_scale_index=source_scale_index,
                    input_path=input_path,
                    device=device,
                    current_filename=filename,
                )
                q_weights[key] = block_quant_dequant(
                    tensor,
                    source_scale_fp32,
                    SOURCE_FP8_BLOCK_SIZE,
                    torch.bfloat16,
                ).contiguous()
            else:
                q_weights[key] = tensor

    for key, tensor in weights.items():
        if key.endswith(".weight"):
            continue
        # For source FP8 checkpoints, do not copy original block-scale tensors.
        if source_is_block_fp8_ue8m0 and key.endswith(SOURCE_FP8_SCALE_KEY_SUFFIX):
            continue
        q_weights[key] = tensor

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_mxfp8(
    model_dir: str,
    save_dir: str,
    device: str,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    extra_high_precision_layers_hf: tuple[str, ...] = (),
) -> None:
    # [FUNCTION DOC] Converts a whole model directory. Reads the config (layer count), decides
    # the source type (block FP8 ue8m0, or BF16-like; otherwise raises a value error), copies
    # non-safetensors files, reads the weight index to get the sorted file list and (for FP8
    # sources) a map from scale tensor names to files, processes files one at a time freeing
    # memory between, then writes a config with a dynamic e4m3 mxfp8 quantization section (block
    # size 1 by 32, ue8m0 scales, naturally sorted deduplicated unconverted modules if any) and
    # a new weight index with total byte size. Requires an index file in the input.
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(input_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    num_hidden_layers = int(cfg["num_hidden_layers"])
    if _is_source_block_fp8_ue8m0_checkpoint(cfg):
        source_is_block_fp8_ue8m0 = True
    elif _is_bf16_source_checkpoint(cfg):
        source_is_block_fp8_ue8m0 = False
    else:
        raise ValueError(
            "Unsupported source quantization_config. "
            "Only BF16/FP16/FP32 sources and "
            "{quant_method=fp8, weight_block_size=[128, 128], scale_fmt=ue8m0} sources are supported."
        )

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    index_path = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    safetensors_files = sorted(set(weight_map.values()))
    source_scale_index: dict[str, str] = {}
    if source_is_block_fp8_ue8m0:
        source_scale_index = {
            key: filename for key, filename in weight_map.items() if key.endswith(SOURCE_FP8_SCALE_KEY_SUFFIX)
        }

    result_collector = ConversionResult()
    for filename in tqdm(safetensors_files, desc="Processing files"):
        process_file(
            input_path,
            output_path,
            filename,
            result_collector,
            device,
            num_hidden_layers,
            num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16,
            source_is_block_fp8_ue8m0,
            extra_high_precision_layers_hf,
            source_scale_index,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "mxfp8",
        "weight_block_size": TARGET_MXFP8_BLOCK_SIZE,
        "scale_fmt": "ue8m0",
    }
    if len(result_collector.modules_to_not_convert) > 0:

        def natural_key(s):
            # [FUNCTION DOC] HELPER: Sort key splitting a string into digit runs (as integers)
            # and non-digit runs, for natural ordering of layer indices.
            return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]

        quantization_config["modules_to_not_convert"] = sorted(
            list(set(result_collector.modules_to_not_convert)), key=natural_key
        )

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    index_dict = {
        "weight_map": result_collector.weight_map,
        "metadata": {"total_size": result_collector.total_size},
    }
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses known options (model dir, save dir, device default
    # "cuda", counts of layers kept in BF16 at start and end, extra skip substrings; unknown
    # options ignored). Raises a runtime error without CUDA or if the device is not CUDA; a bare
    # digit device means that GPU index; no index means GPU 0. Sets the device, creates the save
    # directory (value error if it is a non-directory) and runs the conversion with stripped,
    # non-empty extra substrings.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HF safetensors model.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save converted model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run quantization on (default: cuda).",
    )
    parser.add_argument(
        "--num-layers-at-start-in-bf16",
        type=int,
        default=0,
        help="Keep first N decoder layers in BF16 and do not quantize them.",
    )
    parser.add_argument(
        "--num-layers-at-end-in-bf16",
        type=int,
        default=0,
        help="Keep last N decoder layers in BF16 and do not quantize them.",
    )
    parser.add_argument(
        "--extra-high-precision-layers-hf",
        type=str,
        nargs="*",
        default=(),
        help="Extra substrings for weight names to skip quantization (e.g. .shared_experts.).",
    )
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run MXFP8 quantization.")

    if isinstance(args.device, str) and args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("MXFP8 quantization requires a CUDA device.")
    if device.index is None:
        device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_mxfp8(
        args.model_dir,
        args.save_dir,
        str(device),
        num_layers_at_start_in_bf16=args.num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=args.num_layers_at_end_in_bf16,
        extra_high_precision_layers_hf=tuple(
            s.strip() for s in args.extra_high_precision_layers_hf if isinstance(s, str) and s.strip()
        ),
    )


if __name__ == "__main__":
    main()
