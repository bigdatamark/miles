# [MODULE DOC] Command-line tool converting routed-expert weights of a Hugging Face safetensors
# checkpoint to NVFP4 (packed fp4 with 1x16 block scales plus a global scale). Only expert
# projection weights inside the main language decoder layers are quantized; shared experts,
# dense layers, extra layers (appended multi-token-prediction layers beyond the configured
# count), vision/audio parts, optionally the first/last N layers and user-given substrings stay
# unchanged. Gate and up projections of one expert group are quantized jointly (pair routine,
# shared global scale) even when they live in different files, using deferred output entries.
# Also writes a ModelOpt-style quantization config into config.json and a separate quant config
# file, with an ignore list augmented with fused-qkv and expert container names, and a new
# weight index. The input scale is stored as ones. Runs on CUDA by default.
"""
python tools/convert_hf_to_nvfp4.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR]
                                   [--device DEVICE]
                                   [--num-layers-at-start-in-bf16 NUM_LAYERS_AT_START_IN_BF16]
                                   [--num-layers-at-end-in-bf16 NUM_LAYERS_AT_END_IN_BF16]
                                   [--extra-high-precision-layers-hf ...]

Convert a BF16/FP16/FP32 HF safetensors checkpoint to NVFP4 (E2M1) for MoE
routed-expert GEMMs in the main language decoder only. MTP, vision/audio
components, shared experts, and dense linear layers are left unmodified.
Use --extra-high-precision-layers-hf to keep additional HF weight-name
substrings unquantized.

This follows Transformer Engine NVFP4 quantization and uses 1D block scaling
(NVTE_NVFP4_1D_SCALING, group size = 16).
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
from tqdm import tqdm

from miles.utils.nvfp4 import NVFP4_GROUP_SIZE, nvfp4_quantize_1d, nvfp4_quantize_1d_pair

DEFAULT_KV_CACHE_SCHEME = {"dynamic": False, "num_bits": 8, "type": "float"}
DEFAULT_KV_CACHE_QUANT_ALGO = "FP8"

EXPERT_WEIGHT_SUFFIXES = (
    ".w1.weight",
    ".w2.weight",
    ".w3.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
)

ROUTED_EXPERT_NAME_MARKERS = (
    ".experts.",
    "block_sparse_moe.experts.",
    ".moe.experts.",
)

SHARED_EXPERT_NAME_MARKERS = (
    ".shared_expert.",
    ".shared_experts.",
)

LANGUAGE_MODEL_LAYER_PREFIXES = (
    "model.layers",
    "language_model.model.layers",
    "model.language_model.layers",
    "language_model.layers",
)
LANGUAGE_MODEL_LAYER_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(prefix) for prefix in LANGUAGE_MODEL_LAYER_PREFIXES) + r")\.(\d+)\."
)

FUSED_QKV_SUFFIXES = (".q_proj", ".k_proj", ".v_proj")
GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}


def _is_routed_expert_weight_name(name: str) -> bool:
    # [FUNCTION DOC] HELPER: True for a weight name that ends with ".weight", is not a shared
    # expert, contains a routed-expert marker and ends with a known expert projection suffix.
    if not name.endswith(".weight"):
        return False
    if any(marker in name for marker in SHARED_EXPERT_NAME_MARKERS):
        return False
    if not any(marker in name for marker in ROUTED_EXPERT_NAME_MARKERS):
        return False
    return any(name.endswith(suffix) for suffix in EXPERT_WEIGHT_SUFFIXES)


def _get_num_hidden_layers(model_dir: str) -> int:
    # [FUNCTION DOC] HELPER: Reads config.json in the directory (value error if missing) and
    # returns the layer count, preferring the nested text config's value over the top-level one;
    # raises a value error if neither exists.
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError("config.json is required to identify the main language decoder layers.")
    cfg = json.load(open(config_path))
    text_config = cfg.get("text_config")
    num_layers = text_config.get("num_hidden_layers") if isinstance(text_config, dict) else None
    if num_layers is None:
        num_layers = cfg.get("num_hidden_layers")
    if num_layers is None:
        raise ValueError("num_hidden_layers not found in config.json.")
    return int(num_layers)


def _get_bf16_layer_prefixes(num_hidden_layers: int, num_at_start: int, num_at_end: int) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Builds the sorted tuple of name prefixes (each known layer prefix,
    # a dot, layer index, trailing dot) for the first and last given numbers of layers.
    layer_indices = set(range(num_at_start)) | set(range(num_hidden_layers - num_at_end, num_hidden_layers))
    return tuple(sorted(f"{prefix}.{i}." for prefix in LANGUAGE_MODEL_LAYER_PREFIXES for i in layer_indices))


def should_quantize(
    name: str,
    weight: torch.Tensor,
    skip_weight_substrings: tuple[str, ...] = (),
    *,
    num_hidden_layers: int | None = None,
) -> bool:
    # [FUNCTION DOC] Decides whether a tensor is quantized: false unless the name belongs to a
    # language-model decoder layer (index below the layer count when given), has no skip
    # substring, is a routed expert weight, is a 16/32-bit float and has at least two dims.
    # Raises a value error if all other conditions hold but the last dim is not divisible by 16.
    # An allowlist keeps modality towers and separately named MTP stacks out.
    # DeepSeek/GLM checkpoints append MTP to model.layers, so also bound its index.
    match = LANGUAGE_MODEL_LAYER_PATTERN.match(name)
    if match is None or (num_hidden_layers is not None and int(match.group(1)) >= num_hidden_layers):
        return False
    if any(substr in name for substr in skip_weight_substrings):
        return False
    if not _is_routed_expert_weight_name(name):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"Last dim {weight.shape[-1]} must be divisible by {NVFP4_GROUP_SIZE} " f"for NVFP4 quantization ({name})."
        )
    return True


def _quantize_nvfp4_1d(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Makes the weight contiguous, checks the last dim is divisible by 16
    # (value error otherwise) and delegates to the shared 1-D quantizer, returning its triple.
    """
    NVFP4 1D quantization (tile shape = 1x16).

    Returns:
      qweight: uint8 packed fp4, shape (M, K // 2)
      block_scale: float8_e4m3fn, shape (M, K // 16)
      global_scale: float32 scalar tensor
    """
    weight = weight.contiguous()
    _, n = weight.shape
    if n % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 requires K divisible by {NVFP4_GROUP_SIZE}, got {n}.")

    return nvfp4_quantize_1d(weight)


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Quantizes a 2-D weight directly, or a 3-D stack (experts first) one slice at
    # a time with results stacked along a new first axis; other ranks raise a value error.
    # Returns (packed weight, block scale, global scale).
    if weight.dim() == 2:
        return _quantize_nvfp4_1d(weight)
    if weight.dim() == 3:
        qweights = []
        block_scales = []
        global_scales = []
        for idx in range(weight.shape[0]):
            qweight, block_scale, global_scale = _quantize_nvfp4_1d(weight[idx])
            qweights.append(qweight)
            block_scales.append(block_scale)
            global_scales.append(global_scale)
        return (
            torch.stack(qweights, dim=0),
            torch.stack(block_scales, dim=0),
            torch.stack(global_scales, dim=0),
        )
    raise ValueError(f"Unsupported weight rank {weight.dim()} for NVFP4 quantization.")


class ConversionResult:
    # [CLASS DOC] HELPER: Accumulates tensor-to-file map, total byte size and unconverted module
    # names across files (sequential use, no locking).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Empty map, zero size, empty list.
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(self, filename: str, q_weights: dict[str, torch.Tensor], module_names: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Records each tensor's file, adds byte sizes, extends module list.
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def _update_quantization_config(cfg: dict, ignore_list: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Mutates a model config mapping in place. Ensures a quantization
    # section with algorithm NVFP4, method modelopt, group size 16, the ignore list, and a
    # default kv-cache scheme if missing; for each existing config group sets default targets
    # (Linear) and, for input-activation and weight sections, default non-dynamic 4-bit float and
    # group size 16. Recurses into a nested text config that carries its own quantization section.
    quant_cfg = cfg.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        quant_cfg = {}

    quant_cfg["quant_algo"] = "NVFP4"
    quant_cfg["quant_method"] = "modelopt"
    quant_cfg["group_size"] = NVFP4_GROUP_SIZE
    quant_cfg["ignore"] = ignore_list
    quant_cfg.setdefault("kv_cache_scheme", DEFAULT_KV_CACHE_SCHEME)

    config_groups = quant_cfg.get("config_groups")
    if isinstance(config_groups, dict):
        for group in config_groups.values():
            if not isinstance(group, dict):
                continue
            group.setdefault("targets", ["Linear"])
            for key in ("input_activations", "weights"):
                section = group.get(key)
                if not isinstance(section, dict):
                    continue
                section.setdefault("dynamic", False)
                section.setdefault("num_bits", 4)
                section.setdefault("type", "float")
                section["group_size"] = NVFP4_GROUP_SIZE

    cfg["quantization_config"] = quant_cfg
    # Composite model configs can expose the text config's quantization policy
    # as their top-level policy when loaded. Do not leave that policy stale.
    text_config = cfg.get("text_config")
    if isinstance(text_config, dict) and "quantization_config" in text_config:
        _update_quantization_config(text_config, ignore_list)


def _write_hf_quant_config(output_path: str, ignore_list: list[str], input_path: str) -> None:
    # [FUNCTION DOC] HELPER: Writes hf_quant_config.json in the output directory, starting from
    # the input's file of that name if present (else a ModelOpt producer stub), setting the
    # quantization section to NVFP4 with FP8 kv cache algorithm, group size 16 and the ignore
    # list as excluded modules.
    hf_quant_path = os.path.join(input_path, "hf_quant_config.json")
    if os.path.exists(hf_quant_path):
        with open(hf_quant_path) as f:
            hf_quant_cfg = json.load(f)
    else:
        hf_quant_cfg = {"producer": {"name": "modelopt"}}

    quant_section = hf_quant_cfg.get("quantization")
    if not isinstance(quant_section, dict):
        quant_section = {}

    quant_section["quant_algo"] = "NVFP4"
    quant_section["kv_cache_quant_algo"] = DEFAULT_KV_CACHE_QUANT_ALGO
    quant_section["group_size"] = NVFP4_GROUP_SIZE
    quant_section["exclude_modules"] = ignore_list
    hf_quant_cfg["quantization"] = quant_section

    with open(os.path.join(output_path, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_cfg, f, indent=2)


def _augment_ignore_list(ignore_list: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the sorted deduplicated ignore list plus extras: for a name
    # ending in a separate q, k or v projection, the fused qkv projection name of the same parent;
    # for a name of the form experts (optionally an expert index) followed by a projection name,
    # the experts container name.
    ignore_set = set(ignore_list)
    extra = set()
    for name in ignore_list:
        if name.endswith(FUSED_QKV_SUFFIXES):
            for suffix in FUSED_QKV_SUFFIXES:
                if name.endswith(suffix):
                    extra.add(name[: -len(suffix)] + ".qkv_proj")
                    break
        match = re.fullmatch(r"(.*\.experts)(?:\.\d+)?\.(gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3)", name)
        if match:
            extra.add(match.group(1))
    ignore_set.update(extra)
    return sorted(ignore_set)


def _split_gated_pair_name(name: str) -> tuple[str | None, str | None]:
    # [FUNCTION DOC] HELPER: If the name ends with a gate or up projection suffix (gate_proj, w1
    # for gate; up_proj, w3 for up), returns (name without the suffix, "gate" or "up"); else
    # (none, none).
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def _collect_gated_pair_locations(
    *,
    input_path: str,
    safetensors_files: list[str],
    device: str,
    skip_weight_substrings: tuple[str, ...],
    num_hidden_layers: int,
) -> dict[str, dict[str, tuple[str, str]]]:
    # [FUNCTION DOC] HELPER: Pre-scan of all files to locate gate/up weight pairs. Loads every
    # tensor of every file (slow, memory heavy on the device), keeps those that would be
    # quantized and have a gate/up suffix, and records for each base name the file and key of
    # each role; a duplicate role raises a value error. Returns only bases with both roles,
    # mapping base to a mapping role to (file name, tensor key).
    gated_pairs: dict[str, dict[str, tuple[str, str]]] = {}
    for filename in safetensors_files:
        with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if not should_quantize(
                    key, tensor, skip_weight_substrings=skip_weight_substrings, num_hidden_layers=num_hidden_layers
                ):
                    continue
                base, role = _split_gated_pair_name(key)
                if base is None or role is None:
                    continue
                roles = gated_pairs.setdefault(base, {})
                if role in roles:
                    raise ValueError(
                        f"NVFP4 requires a single complete gate/up pair per converted checkpoint; "
                        f"found duplicate {role} tensor for {base}."
                    )
                roles[role] = (filename, key)
    return {base: roles for base, roles in gated_pairs.items() if set(roles) == {"gate", "up"}}


def _nvfp4_quantized_entries(
    key: str,
    qweight: torch.Tensor,
    block_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Returns the four checkpoint entries for one quantized weight: the
    # packed weight under the original key, block scale as ".weight_scale", global scale as
    # ".weight_scale_2", and an input scale of ones (float32, same shape as the global scale)
    # as ".input_scale".
    return {
        key: qweight,
        key.replace(".weight", ".weight_scale"): block_scale,
        key.replace(".weight", ".weight_scale_2"): weight_scale_2,
        key.replace(".weight", ".input_scale"): torch.ones_like(weight_scale_2, dtype=torch.float32),
    }


def _load_safetensors_tensor(input_path: str, filename: str, key: str, device: str) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Opens the named file and returns the one tensor on the device.
    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        return f.get_tensor(key)


def process_file(
    input_path: str,
    output_path: str,
    filename: str,
    result_collector: ConversionResult,
    device: str,
    num_hidden_layers: int,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
    extra_high_precision_layers_hf: tuple[str, ...],
    gated_pair_locations: dict[str, dict[str, tuple[str, str]]],
    processed_gated_pairs: set[str],
    deferred_quantized_entries: dict[str, dict[str, dict[str, torch.Tensor]]],
) -> None:
    # [FUNCTION DOC] Converts one file (returns at once for non-safetensors names). Builds skip
    # substrings from extra substrings and BF16 layer prefixes. For each tensor key: if entries
    # were deferred for it by a pair processed from another file, uses them; else loads it and,
    # if quantizable: for a gate/up pair, loads the partner (from another file if needed),
    # quantizes the pair together, stores the entries for this file's key now and defers the
    # partner's entries when it lives in another file (raises a value error if the pair was
    # already processed but nothing deferred); otherwise quantizes alone. Unquantized weights are
    # kept and their modules recorded as unconverted (matched BF16 layer prefixes, the module
    # name; for experts only the experts container when inside a BF16 layer, else the module).
    # Saves the file and reports. Mutates the shared pair-tracking containers.
    if not filename.endswith(".safetensors"):
        return

    modules_to_not_convert: list[str] = []
    q_weights: dict[str, torch.Tensor] = {}
    dynamic_skip_layer_prefixes = _get_bf16_layer_prefixes(
        num_hidden_layers, num_layers_at_start_in_bf16, num_layers_at_end_in_bf16
    )

    dynamic_skip_substrings = (
        *extra_high_precision_layers_hf,
        *sorted(dynamic_skip_layer_prefixes),
    )

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        for key in f.keys():
            if key in q_weights:
                continue
            deferred_entries = deferred_quantized_entries.get(filename, {}).pop(key, None)
            if deferred_entries is not None:
                q_weights.update(deferred_entries)
                continue

            tensor = f.get_tensor(key)
            if should_quantize(
                key, tensor, skip_weight_substrings=dynamic_skip_substrings, num_hidden_layers=num_hidden_layers
            ):
                base, _role = _split_gated_pair_name(key)
                if base in gated_pair_locations:
                    if base in processed_gated_pairs:
                        raise ValueError(f"Missing deferred NVFP4 output for already processed gated pair {base}.")

                    pair = gated_pair_locations[base]
                    gate_filename, gate_key = pair["gate"]
                    up_filename, up_key = pair["up"]
                    gate_weight = (
                        tensor
                        if key == gate_key
                        else _load_safetensors_tensor(input_path, gate_filename, gate_key, device)
                    )
                    up_weight = (
                        tensor if key == up_key else _load_safetensors_tensor(input_path, up_filename, up_key, device)
                    )
                    gate_output, up_output = nvfp4_quantize_1d_pair(gate_weight, up_weight)
                    for target_filename, target_key, output in (
                        (gate_filename, gate_key, gate_output),
                        (up_filename, up_key, up_output),
                    ):
                        entries = _nvfp4_quantized_entries(target_key, *output)
                        if target_filename == filename:
                            q_weights.update(entries)
                        else:
                            deferred_quantized_entries.setdefault(target_filename, {})[target_key] = entries
                    processed_gated_pairs.add(base)
                    continue

                qweight, block_scale, weight_scale_2 = quantize_nvfp4(tensor)
                q_weights.update(_nvfp4_quantized_entries(key, qweight, block_scale, weight_scale_2))
            else:
                if key.endswith(".weight"):
                    module_name = key[: -len(".weight")]
                    matched_layer_prefixes = [
                        prefix for prefix in dynamic_skip_layer_prefixes if key.startswith(prefix)
                    ]
                    modules_to_not_convert.extend(matched_layer_prefixes)
                    if ".experts." not in key:
                        modules_to_not_convert.append(module_name)
                    elif matched_layer_prefixes:
                        # ModelOpt needs the exact FusedMoE container in addition to the layer prefix.
                        expert_prefix = module_name.split(".experts.", 1)[0] + ".experts"
                        modules_to_not_convert.append(expert_prefix)
                    else:
                        modules_to_not_convert.append(module_name)
                q_weights[key] = tensor

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_nvfp4(
    model_dir: str,
    save_dir: str,
    device: str,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    extra_high_precision_layers_hf: tuple[str, ...] = (),
) -> None:
    # [FUNCTION DOC] Converts a whole model directory: copies non-safetensors files, determines
    # the layer count, pre-scans gate/up pair locations, processes each safetensors file in
    # listing order (freeing GPU cache between), raises a runtime error if any deferred pair
    # entries were never written, builds the augmented ignore list, writes the updated config (if
    # present), the separate quant config file and a new weight index with total byte size.
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    num_hidden_layers = _get_num_hidden_layers(input_path)
    dynamic_skip_layer_prefixes = _get_bf16_layer_prefixes(
        num_hidden_layers, num_layers_at_start_in_bf16, num_layers_at_end_in_bf16
    )
    dynamic_skip_substrings = (
        *extra_high_precision_layers_hf,
        *sorted(dynamic_skip_layer_prefixes),
    )

    gated_pair_locations = _collect_gated_pair_locations(
        input_path=input_path,
        safetensors_files=safetensors_files,
        device=device,
        skip_weight_substrings=dynamic_skip_substrings,
        num_hidden_layers=num_hidden_layers,
    )
    processed_gated_pairs: set[str] = set()
    deferred_quantized_entries: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
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
            extra_high_precision_layers_hf,
            gated_pair_locations,
            processed_gated_pairs,
            deferred_quantized_entries,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    remaining_deferred_entries = {
        filename: sorted(entries) for filename, entries in deferred_quantized_entries.items() if entries
    }
    if remaining_deferred_entries:
        raise RuntimeError(f"Unwritten deferred NVFP4 gated-pair entries: {remaining_deferred_entries}")

    ignore_list = _augment_ignore_list(result_collector.modules_to_not_convert)

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        _update_quantization_config(cfg, ignore_list)
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    _write_hf_quant_config(output_path, ignore_list, input_path)

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
    # "cuda", BF16 start and end layer counts, extra skip substrings; unknown options ignored).
    # A digit-only device means that GPU index. For a CUDA device: raises a runtime error if CUDA
    # is unavailable, defaults to GPU 0, sets it. Creates the save directory (value error if the
    # path is not a directory) and runs the conversion with stripped, non-empty substrings.
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
        default=[],
        help="Additional HF weight-name substrings to keep unquantized.",
    )
    args, _ = parser.parse_known_args()

    if isinstance(args.device, str) and args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run NVFP4 quantization.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_nvfp4(
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
