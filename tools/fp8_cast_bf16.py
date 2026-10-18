# Adapt from https://github.com/alibaba/Pai-Megatron-Patch/blob/2b201af08336dea0403df7c6b497c964cf5a2e75/toolkits/model_checkpoints_convertor/deepseek/fp8_cast_bf16.py
# [MODULE DOC] Command-line tool dequantizing a block-FP8 Hugging Face checkpoint (DeepSeek
# style: weights stored as 1-byte values with a companion tensor named like the weight plus
# "_scale_inv", one scale per 128 by 128 block) to BF16 on the GPU. Tensor names are remapped
# through the parameter-name remap helper (sibling module imported by bare name, so the script
# must run with this directory on the module path). Copies config, python files, tokenizer and
# chat template files with shell commands (unquoted paths), converts each safetensors file, and
# writes a new index without the dequantized weights' scale entries. Options: input FP8 dir,
# output BF16 dir. Sets the global default dtype to bfloat16 (affects dequantized output dtype).
import json
import os
from argparse import ArgumentParser
from glob import glob

import torch
import triton
import triton.language as tl
from param_name_remap import get_param_name_remap
from safetensors.torch import load_file, save_file
from tqdm import tqdm


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # [FUNCTION DOC] Triton kernel; one program per square tile (grid over rows and columns of
    # tiles). Loads the tile of the M by N row-major quantized matrix (masked at the edges) as
    # float32, multiplies by the tile's single scale (scale grid indexed by tile row times the
    # number of column tiles plus tile column), and stores to the output buffer in its dtype.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    # [FUNCTION DOC] Dequantizes a 2-D contiguous quantized matrix with a contiguous 2-D scale
    # grid (one per block of the given size, default 128) using the Triton kernel. The output
    # has the current default dtype (set to bfloat16 by the main routine), same shape as the
    # input. Asserts contiguity and rank.
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta):
        # [FUNCTION DOC] HELPER: Launch grid: number of row tiles and column tiles for the block
        # size chosen in the kernel meta-parameters.
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def main(fp8_path, bf16_path):
    # [FUNCTION DOC] Runs the conversion. Sets default dtype bfloat16, creates the output dir,
    # copies metadata files by shell command, reads the weight index and builds the remapped
    # weight map plus a reverse map from new to raw names (asserting no collisions). For each
    # safetensors file in sorted order: loads it to the GPU, caches it; for each tensor, remaps
    # its name; scale tensors (name ending "_scale_inv") are skipped; 1-byte tensors are
    # dequantized with the scale looked up (possibly from another file) after conversion to
    # float32, or kept unchanged with a warning when the scale is missing (KeyError); other
    # tensors are copied. Saves the file under the same name; evicts the oldest cached file when
    # more than two are cached. Finally removes scale entries of converted weights from the
    # remapped map and writes the index (empty metadata).
    # KNOWN ISSUE: the oldest-cached eviction can drop the file just processed or still needed,
    # and a scale tensor present only in a not-yet-loaded file is reloaded on demand.
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    os.system("cp -rf " + fp8_path + "/config.json " + bf16_path)
    os.system("cp -rf " + fp8_path + "/*.py " + bf16_path)
    os.system("cp -rf " + fp8_path + "/tokenizer* " + bf16_path)
    os.system("cp -rf " + fp8_path + "/chat_template* " + bf16_path)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map_raw = model_index["weight_map"]

    remap = get_param_name_remap(os.path.join(fp8_path, "config.json"), weight_map_raw)
    weight_map_renamed = {}
    raw_name_by_renamed = {}
    for raw_name, file_name in weight_map_raw.items():
        renamed_name = remap(raw_name)
        assert renamed_name not in raw_name_by_renamed, (
            f"Remapped tensor name collision: {renamed_name} from "
            f"{raw_name} and {raw_name_by_renamed[renamed_name]}"
        )
        weight_map_renamed[renamed_name] = file_name
        raw_name_by_renamed[renamed_name] = raw_name

    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        # [FUNCTION DOC] HELPER: Returns the tensor for a remapped name from the file the index
        # says holds it, loading and caching that file on the GPU if needed. Raises a key error if
        # the name is unknown.
        raw_tensor_name = raw_name_by_renamed[tensor_name]
        file_name = weight_map_raw[raw_tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")

        return loaded_files[file_name][raw_tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        print(f"Handling file: {safetensor_file}")
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name_raw, weight in current_state_dict.items():
            weight_name = remap(weight_name_raw)

            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
                else:
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv.float())
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map_renamed:
            weight_map_renamed.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map_renamed}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
