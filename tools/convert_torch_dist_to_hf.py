# [MODULE DOC] Single-process command-line tool converting a Megatron distributed checkpoint to
# a Hugging Face safetensors model, without GPUs or a process group. Loads the whole checkpoint
# on CPU (skipping optimizer entries and state entries), prefixes names with the Megatron module
# wrapper prefix, expands stacked layer tensors into per-layer tensors and stacked expert tensors
# into per-expert tensors, optionally strips vocabulary padding, converts to HF names with the
# shared conversion routines, packs outputs into chunks of at most the chunk size, writes the
# index first and then the numbered safetensors files, and optionally copies non-weight assets
# from an original HF directory. Importing this module replaces the global pickle unpickler class
# with the tolerant one (side effect). Options: model name (or derived from the original HF
# config class name), input dir, output dir, original HF dir, force, chunk size (default 5 GiB),
# vocab size.
import argparse
import json
import os
import pickle
import re
import shutil
import time

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding
from miles.utils.hf_utils.config import load_hf_config


class UnpicklerWrapper(pickle.Unpickler):
    # [CLASS DOC] HELPER: Unpickler substituting a placeholder class for classes from modules
    # starting with megatron or glm, so checkpoint metadata loads without those packages.
    @override
    def find_class(self, mod_name, name):
        # [FUNCTION DOC] HELPER: Returns the placeholder (accepts any constructor arguments) for
        # megatron/glm modules, else the normal lookup.
        class DummyClass:
            # [CLASS DOC] HELPER: Placeholder ignoring all constructor arguments.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Accepts and ignores all arguments.
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    # [CLASS DOC] HELPER: Filesystem checkpoint reader tolerant of missing Megatron/GLM classes
    # and of metadata without storage info or planner data.
    @override
    def read_metadata(self):
        # [FUNCTION DOC] HELPER: Loads the hidden metadata file with the tolerant unpickler;
        # creates empty storage metadata when absent, sets its load id from the reader, and
        # replaces missing planner data by an empty mapping. Returns the metadata.
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
    # [CLASS DOC] HELPER: Load planner that creates the destination tensors from checkpoint
    # metadata, so the whole checkpoint loads into an initially empty mapping.
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        # [FUNCTION DOC] HELPER: For every metadata entry whose key contains neither "optimizer"
        # nor "_state", prints a found message and stores an uninitialized CPU tensor of the
        # recorded size and dtype (tensor entries) or the metadata value itself (others) in the
        # state dict; then runs the parent setup. Metadata is required in practice.
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def get_expert_param(args, name, param):
    # [FUNCTION DOC] HELPER: Generator of (name, tensor) pairs. Names without ".experts." pass
    # through. For an expert tensor whose name lacks a trailing expert index after "weight"
    # (stacked experts; asserts the first dim equals the configured expert count), yields one
    # pair per expert: the name with the doubled "experts.experts." collapsed and the expert
    # index appended, and that expert's slice. Names that already carry the index pass through.
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            expert_param = param[expert_id]
            yield expert_name, expert_param
    else:
        yield name, param


def get_layer_param(args, name, param):
    # [FUNCTION DOC] HELPER: Generator of (name, tensor) pairs. Names without ".layers." pass
    # through. A layers name with no layer index (stacked layers; asserts first dim equals the
    # configured layer count) is expanded into one entry per layer, with the layer index inserted
    # after ".layers.", each passed on to the expert expansion. Names with an index go straight
    # to the expert expansion.
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            layer_param = param[layer_id]
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    # [FUNCTION DOC] HELPER: Generator over the loaded state dict, prefixing each name with the
    # doubled module wrapper prefix and yielding the layer- and expert-expanded pairs.
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None):
    # [FUNCTION DOC] Converts and writes all tensors. Sets the expert-parallel MoE flag on the
    # args to false (mutation, needed for compatibility). Creates the output directory. For each
    # expanded parameter: strips vocabulary padding when a vocab size is given, converts to HF
    # (name, tensor) pairs, and appends to the current chunk, starting a new chunk when adding
    # would exceed the chunk size in bytes (an oversized single tensor gets its own chunk; an
    # empty first chunk can result if the first tensor alone exceeds the chunk size). Writes
    # the index (total size and weight map to files named model-IIIII-of-NNNNN.safetensors)
    # before the data files, then saves each file with timing output.
    # KNOWN ISSUE: the chunk-size help text and an inline comment say 2GB but the default is 5 GiB.
    # for miles update_weight compatible
    args.sglang_enable_ep_moe = False

    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # 2GB
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
            total_size += tensor_size

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

    num_files = len(modeltensors)
    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    print(f"{index_filepath} saved.")

    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        filepath = os.path.join(output_dir, filename)
        safetensors.torch.save_file(tensors, filepath)
        print(f"{filename} saved in {time.time() - t:.2f} sec.")


def copy_assets(origin_hf_dir, output_dir):
    # [FUNCTION DOC] Copies every regular file from the original HF directory to the output
    # directory except safetensors files and the safetensors index; directories are skipped with
    # a message. Prints each copy; overwrites.
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
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = load_hf_config(args.origin_hf_dir)
        args.model_name = type(hf_config).__name__.lower()

    state_dict = {}
    print(f"loading model from {args.input_dir}")
    t = time.time()
    megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
    if args.origin_hf_dir:
        megatron_args.hf_checkpoint = args.origin_hf_dir
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(args.input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"model loaded in {time.time()-t:.2f} sec.")

    save_tensors(megatron_args, args.model_name, state_dict, args.output_dir, args.chunk_size, args.vocab_size)

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
