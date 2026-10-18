# [MODULE DOC] Command-line tool quantizing a Hugging Face safetensors model to FP8 (e4m3) on
# GPU. Strategies: block (per tile scales, default), channel (per output row) or tensor.
# Every safetensors file is processed (thread pool, default one worker); weights whose names
# match the exclusion list (norms, embeddings, routers, heads, vision parts, etc.) are copied
# unquantized. Non-weight files are copied, then config gets a quantization section and a new
# weight index is written. Needs CUDA.
"""
python tools/convert_hf_to_fp8.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR] [--strategy {block,channel,tensor}] [--block-size [BLOCK_SIZE ...]]
                           [--max-workers MAX_WORKERS]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Path to the directory of the HF safetensors model.
  --save-dir SAVE_DIR   Path to the directory to save the converted model.
  --strategy {block,channel,tensor}
  --block-size [BLOCK_SIZE ...]
                        eg. --block-size 32 32
  --max-workers MAX_WORKERS
                        Number of worker threads for parallel processing
"""

import argparse
import gc
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from tqdm import tqdm

FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_MAX, FP8_MIN = FP8_INFO.max, FP8_INFO.min


def ceildiv(a, b):
    # [FUNCTION DOC] HELPER: Integer division rounding up (via negated floor division).
    return -(-a // b)


def block_fp8(weight, block_size):
    # [FUNCTION DOC] Block-wise quantization of a 2-D weight with tile sizes (rows, columns).
    # Zero-pads to whole tiles, finds each tile's max absolute value, scale = max / FP8 max
    # (float32), divides, clamps to the FP8 range and casts, crops back to the original shape.
    # Returns (quantized weight, scale grid of shape row tiles by column tiles); dequantize by
    # multiplying. An all-zero tile gets scale zero, so division yields not-a-number values.
    # KNOWN ISSUE: no lower bound on the scale (unlike the channel and tensor variants), so a
    # zero tile produces NaN after dividing zero by zero.

    # per block quant
    block_n, block_k = block_size[0], block_size[1]

    shape_0, shape_1 = weight.shape

    n_tiles = ceildiv(shape_0, block_n)
    k_tiles = ceildiv(shape_1, block_k)

    q_weight = F.pad(
        weight,
        (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0),
        mode="constant",
        value=0.0,
    )

    qweight = q_weight.reshape(n_tiles, block_n, k_tiles, block_k)
    block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
    block_max = torch.max(block_max, dim=3, keepdim=True)[0]

    scale = block_max.to(torch.float32) / FP8_MAX
    qweight = (
        (qweight / scale)
        .clamp(min=FP8_MIN, max=FP8_MAX)
        .reshape((n_tiles * block_n, k_tiles * block_k))
        .to(torch.float8_e4m3fn)
    )
    qweight = qweight[:shape_0, :shape_1].clone().detach()
    scale = scale.reshape(n_tiles, k_tiles)

    return qweight, scale


def channel_fp8(weight):
    # [FUNCTION DOC] Per-row quantization: scale = row max absolute value (floored at 1e-12) over
    # FP8 max, as float32 with shape rows by one. Returns (clamped and cast weight, scale).
    channel_max = torch.max(weight.abs(), dim=-1, keepdim=True)[0]
    scale = channel_max.clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight, scale


def tensor_fp8(weight):
    # [FUNCTION DOC] Whole-tensor quantization: one scale = global max absolute value (floored at
    # 1e-12) over FP8 max, returned with shape of one element. Returns (quantized weight, scale).
    scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.view(1)
    return qweight, scale


def quant_fp8(weight, strategy, block_size=None):
    # [FUNCTION DOC] Dispatches on the strategy name: "tensor", "channel", anything else is
    # treated as block (block size then must be given, else indexing none fails).
    if strategy == "tensor":
        return tensor_fp8(weight)
    elif strategy == "channel":
        return channel_fp8(weight)
    else:
        return block_fp8(weight, block_size)


class ConversionResult:
    # [CLASS DOC] HELPER: Thread-safe accumulator shared by worker threads: weight name to file
    # name map, a count of parameters, list of module names left unquantized, and a flag set when
    # any tensor name contains "indexer" (sparse-attention layers).
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the lock, empty map, zero count, empty list, false flag.
        self.lock = threading.Lock()
        self.weight_map = {}
        self.param_count = 0
        self.modules_to_not_convert = []
        self.has_dsa_layers = False

    def add_result(self, filename, q_weights, module_names):
        # [FUNCTION DOC] HELPER: Under the lock, records each tensor name's file, adds the first
        # dimension length of each tensor to the parameter count, sets the indexer flag, and
        # extends the unquantized module list.
        # KNOWN ISSUE: the count adds only the leading dimension length, not the element count,
        # yet it is written as the total size in the index metadata.
        with self.lock:
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                self.param_count += len(v)
                if "indexer" in k:
                    self.has_dsa_layers = True
            self.modules_to_not_convert.extend(module_names)


def process_file(input_path, output_path, filename, strategy, block_size, result_collector):
    # [FUNCTION DOC] Worker for one file; returns immediately for non-safetensors names. Loads
    # all tensors onto the GPU, quantizes each tensor whose name contains "weight" and none of
    # the excluded substrings (layernorm, embed, router, mlp.gate., norm, lm_head, eh_proj,
    # weights_proj, head., wo_a, ffn.gate., compressor., vision_tower, mm_projector); the scale
    # is stored under the name with ".weight" replaced by ".weight_scale_inv" when a block size
    # is given, else ".weight_scale". Excluded tensors are stored unchanged and their name minus
    # ".weight" is recorded as not converted. Saves the file (format metadata "pt") in the output
    # directory and reports to the collector. Prints GPU memory in use. Holds the full file on GPU.
    if not filename.endswith(".safetensors"):
        return

    print(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    weights = {}
    q_weights = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    modules_to_not_convert = []
    for key in weights.keys():
        if (
            "weight" in key
            and "layernorm" not in key
            and "embed" not in key
            and "router" not in key
            and "mlp.gate." not in key
            and "norm" not in key
            and "lm_head" not in key
            and "eh_proj" not in key
            and "weights_proj" not in key
            and "head." not in key
            and "wo_a" not in key
            and "ffn.gate." not in key
            and "compressor." not in key
            and "vision_tower" not in key
            and "mm_projector" not in key
        ):
            qw, s = quant_fp8(weights[key], strategy, block_size)
            q_weights[key] = qw
            if block_size:
                scale_name = key.replace(".weight", ".weight_scale_inv")
            else:
                scale_name = key.replace(".weight", ".weight_scale")
            q_weights[scale_name] = s
        else:
            modules_to_not_convert.append(key.replace(".weight", ""))
            q_weights[key] = weights[key]

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})

    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_fp8(input_path, output_path, strategy, block_size=None, max_workers=4):
    # [FUNCTION DOC] Converts a whole model directory. Creates the output directory, copies all
    # non-safetensors, non-directory files, processes the safetensors files in a thread pool
    # (re-raising worker errors), then builds the quantization config: for block or tensor
    # strategy a dynamic-activation e4m3 fp8 config, with block size when given, a scale format
    # marker "ue8m0" if sparse-attention indexer tensors were seen (only when a block size is
    # given), and the deduplicated unconverted module list if non-empty; for channel strategy a
    # compressed-tensors config with 8-bit float token-dynamic activations and static weights.
    # Writes it into a copy of config.json if the input has one, and writes a new safetensors
    # index with the weight map. Frees cached GPU memory at the end.
    # KNOWN ISSUE: the config copy and index are written via unclosed file handles.
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(
                process_file, input_path, output_path, filename, strategy, block_size, result_collector
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    if strategy == "block" or strategy == "tensor":
        quantization_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
        }
        if block_size:
            quantization_config["weight_block_size"] = block_size
            if result_collector.has_dsa_layers:
                quantization_config["scale_fmt"] = "ue8m0"
        if len(result_collector.modules_to_not_convert) > 0:
            quantization_config["modules_to_not_convert"] = list(set(result_collector.modules_to_not_convert))
    else:
        quant_group = {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": None,
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "float",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": strategy,
                    "symmetric": True,
                    "type": "float",
                },
            },
        }
        quantization_config = {
            "config_groups": quant_group,
            "format": "float-quantized",
            "ignore": list(set(result_collector.modules_to_not_convert)),
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        }

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    index_dict = {"weight_map": result_collector.weight_map, "metadata": {"total_size": result_collector.param_count}}
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="Path to the directory of the HF safetensors model.")
    parser.add_argument("--save-dir", type=str, help="Path to the directory to save the converted model.")
    parser.add_argument("--strategy", type=str, default="block", choices=["block", "channel", "tensor"])
    parser.add_argument("--block-size", type=int, nargs="*", default=None, help="eg. --block-size 32 32")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker threads for parallel processing")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_fp8(args.model_dir, args.save_dir, args.strategy, args.block_size, args.max_workers)
