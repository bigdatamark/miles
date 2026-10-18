# [MODULE DOC] Command-line tool converting a distributed-checkpoint (FSDP) training checkpoint
# into a Hugging Face safetensors model. Arguments: input checkpoint directory (a model
# subdirectory is used if present), output directory, original Hugging Face directory (for
# config and tokenizer assets), and a force flag. Refuses to run if the output directory exists
# without force. Steps: load all non-optimizer tensors on CPU without a process group, build an
# empty model from the original config, choose the key prefix (among known wrappers) that
# matches the most model keys, load with non-strict matching (prints missing and unexpected
# keys), save as safetensors, then copy every non-weight regular file from the original directory.
import argparse
import os
import pickle
import shutil
import time

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoModelForCausalLM
from typing_extensions import override

from miles.utils.hf_utils.config import load_hf_config


class UnpicklerWrapper(pickle.Unpickler):
    # [CLASS DOC] HELPER: Unpickler that lets checkpoint metadata load without Megatron or GLM
    # packages installed, by substituting a placeholder class for any class from modules whose
    # name starts with megatron or glm; all other classes resolve normally.
    @override
    def find_class(self, mod_name, name):
        # [FUNCTION DOC] HELPER: Returns the placeholder class (accepts any constructor
        # arguments, stores nothing) for megatron/glm modules, else the default lookup.
        class DummyClass:
            # [CLASS DOC] HELPER: Placeholder that ignores all construction arguments.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Accepts and ignores any arguments.
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    # [CLASS DOC] HELPER: Filesystem checkpoint reader whose metadata loading tolerates missing
    # Megatron/GLM classes and older metadata lacking storage info or planner data.
    @override
    def read_metadata(self):
        # [FUNCTION DOC] HELPER: Reads the hidden metadata file in the checkpoint directory with
        # the tolerant unpickler. If storage metadata is absent, creates an empty one; sets its
        # load id from the reader; if planner data is none, sets it to an empty mapping.
        # Returns the metadata object.
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    # [CLASS DOC] HELPER: Load planner that fills the destination state dict from the checkpoint
    # metadata itself, so no model needs to exist before loading.
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        # [FUNCTION DOC] HELPER: For each entry in the metadata (skipping keys containing
        # "optimizer"), prints a found message; tensor entries become uninitialized CPU tensors of
        # the recorded size and dtype, other entries are stored as-is; then calls the parent setup.
        # Mutates the given state dict. The metadata argument is required in practice despite its
        # none default.
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _detect_model_dir(input_dir: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the model subdirectory of the input if it exists, else the
    # input directory itself.
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Loads the whole checkpoint into a new mapping on CPU using the
    # tolerant reader and the metadata-driven planner, without a distributed group (private
    # torch loader API). Returns the mapping from key to tensor (may include non-tensor values).
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _get_candidate_prefixes(keys: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: From the fixed ordered list (model_state.model., model_state.,
    # model., module., empty), returns those that at least one key starts with, in that order;
    # the empty prefix is always included.
    predefined = [
        "model_state.model.",
        "model_state.",
        "model.",
        "module.",
        "",
    ]

    detected: set[str] = set()
    for key in keys:
        for prefix in predefined:
            if prefix and key.startswith(prefix):
                detected.add(prefix)

    # Always keep empty string as a fall back option for exact match.
    detected.add("")
    # Preserve predefined order while keeping only detected prefixes.
    return [p for p in predefined if p in detected]


def _strip_best_prefix(keys: list[str], target_keys: set[str]) -> tuple[str, int]:
    # [FUNCTION DOC] HELPER: For each candidate prefix, strips it from all keys and counts overlap
    # with the target key set; returns (prefix, count) for the strictly best count (ties keep the
    # earliest candidate in order).
    # KNOWN ISSUE: keys lacking the prefix are left unchanged but still counted, and the later
    # conversion strips the prefix from every key the same way.
    best_prefix = ""
    best_match = -1

    for prefix in _get_candidate_prefixes(keys):
        mapped_keys = {k.removeprefix(prefix) for k in keys}
        match_count = len(mapped_keys & target_keys)
        if match_count > best_match:
            best_match = match_count
            best_prefix = prefix

    return best_prefix, best_match


def _convert_fsdp_to_hf(
    origin_hf_dir: str,
    input_dir: str,
    output_dir: str,
) -> None:
    # [FUNCTION DOC] HELPER: Core conversion (see module doc). Loads and times the checkpoint,
    # keeps only tensor values, builds a model from the original config with random weights,
    # picks and prints the best prefix and match count, strips the prefix, raises a value error
    # if no tensors were found, loads non-strictly and prints missing and unexpected keys, then
    # creates the output directory and saves safetensors. Does not copy assets.
    print(f"loading FSDP model from {input_dir}")
    t = time.time()
    state_dict = _load_fsdp_state_dict(input_dir)
    print(f"FSDP model loaded in {time.time()-t:.2f} sec.")

    tensor_items = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    config = load_hf_config(origin_hf_dir)
    hf_model = AutoModelForCausalLM.from_config(config)
    target_keys = set(hf_model.state_dict().keys())

    best_prefix, best_match = _strip_best_prefix(list(tensor_items.keys()), target_keys)
    total_keys = len(tensor_items)

    print(f"Using prefix '{best_prefix}' for key mapping. " f"Matched {best_match}/{total_keys} parameter keys.")

    model_state = {k.removeprefix(best_prefix): v for k, v in tensor_items.items()}

    if not model_state:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )

    missing, unexpected = hf_model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model weights saved to {output_dir}")


def copy_assets(origin_hf_dir: str, output_dir: str) -> None:
    # [FUNCTION DOC] Copies every regular file of the original directory into the output
    # directory except safetensors weight files and the safetensors index; subdirectories are
    # skipped with a message. Prints each copy; overwrites existing files.
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="The original Hugging Face model directory to load config/tokenizer assets.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    model_dir = _detect_model_dir(args.input_dir)
    _convert_fsdp_to_hf(args.origin_hf_dir, model_dir, args.output_dir)
    copy_assets(args.origin_hf_dir, args.output_dir)
