# [MODULE DOC]
# Writes full snapshots of the training state in Hugging Face format to a directory: LoRA adapter
# files (config JSON plus safetensors) and base model shards (numbered safetensors shards, a weight
# index and copied tokenizer/config metadata files). Weights come from the weight iterator, whose
# gathers are collective; only global rank 0 writes files.
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.utils.lora.utils import AdapterSpec, get_adapter_target_modules

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _is_hf_metadata_file(path: Path) -> bool:
    # [FUNCTION DOC] HELPER: True for regular files whose suffix is not a weight-file suffix (the
    # tuple above) and whose name does not end with ".index.json", i.e. config/tokenizer files worth
    # copying from the base checkpoint.
    # The base checkpoint's weight index must not overwrite the exported shard mapping.
    return path.is_file() and path.suffix not in HF_WEIGHT_SUFFIXES and not path.name.endswith(".index.json")


class SnapshotPublisher:
    # [CLASS DOC] Publisher bound to a weight iterator (must gather all parallel dims) and an optional
    # base adapter config mapping (needed only for adapter export). Methods are collective: every
    # rank must call them in lockstep, and only rank 0 writes.
    def __init__(self, iterator: HfWeightIteratorBase, adapter_config: dict | None = None) -> None:
        # [FUNCTION DOC] Asserts the iterator's placement is a full gather; stores the iterator and
        # adapter config.
        assert iterator.placement.is_full_gather, "publishing requires full weights on rank 0"
        self._iterator = iterator
        self._adapter_config = adapter_config

    def publish_adapter(self, adapter: AdapterSpec | None, path: str, metadata: dict | None = None) -> None:
        # [FUNCTION DOC] Publishes an adapter to a new checkpoint directory at the path through the
        # shared checkpoint directory writer (with optional metadata, never overwriting an existing
        # one), whose callback writes the adapter files into the directory. Collective.
        write_checkpoint_dir(
            path,
            lambda checkpoint_dir: self.write_adapter(adapter, checkpoint_dir),
            metadata=metadata,
            overwrite=False,
        )

    @torch.no_grad()
    def write_adapter(self, adapter: AdapterSpec | None, path: str | Path) -> None:
        # [FUNCTION DOC] Runs without autograd. Asserts an adapter config was provided. All ranks
        # call the iterator's adapter materialization (joining collectives); only rank 0 receives the
        # tensors, each made detached, contiguous and on CPU. Rank 0 serializes them to safetensors
        # bytes. The config is the base config, updated with the adapter's rank and alpha as "r" and
        # "lora_alpha" when an adapter spec is given (none keeps the base config values). On rank 0
        # the "target_modules" entry is set from the tensor names, the directory is created (with
        # parents), and the config JSON and the adapter safetensors file are written.
        """Collectively write HF adapter files into the caller's checkpoint directory."""
        assert self._adapter_config is not None, "adapter export requires adapter_config"
        path = Path(path)
        is_writer = dist.get_rank() == 0
        adapter_tensors = {
            name: tensor.detach().contiguous().cpu()
            for name, tensor in self._iterator.materialize_adapter(adapter, materialize=is_writer).items()
        }
        adapter_bytes = safetensors.torch.save(adapter_tensors) if is_writer else None
        adapter_config = self._adapter_config
        if adapter is not None:
            adapter_config = adapter_config | {"r": adapter.rank, "lora_alpha": adapter.alpha}

        if is_writer:
            adapter_config = adapter_config | {"target_modules": get_adapter_target_modules(adapter_tensors)}
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text(json.dumps(adapter_config))
            (path / "adapter_model.safetensors").write_bytes(adapter_bytes)

    def write_model(self, path: str | Path, *, weights: Mapping[str, torch.Tensor], hf_checkpoint: str) -> None:
        # [FUNCTION DOC] Collective. Iterates the iterator's size-bounded buckets over the given
        # backend-native weights (every rank iterates and materializes). Rank 0 writes each bucket as
        # a CPU safetensors shard named "model-<5 digit counter starting at 1>.safetensors" in the
        # directory (which must already exist), recording each tensor's shard and accumulating total
        # byte size. A write failure is remembered, further writing is skipped so the remaining
        # gathers complete on all ranks, then the error is raised at the end. After a clean pass,
        # rank 0 asserts at least one weight was written, copies non-weight regular files (configs,
        # tokenizer files; not index files) from the base checkpoint directory (logs a warning
        # instead if it is not a local directory), and writes the index JSON with total size and
        # weight map (indented).
        """Collectively write HF model shards into the caller's checkpoint directory."""
        path = Path(path)
        is_writer = dist.get_rank() == 0

        weight_map: dict[str, str] = {}
        total_size = 0
        shard_index = 0
        write_error = None
        for hf_named_tensors in self._iterator.iter_hf_weights(weights):
            if not is_writer or write_error is not None:
                continue
            shard_index += 1
            shard_name = f"model-{shard_index:05d}.safetensors"
            shard_tensors = {}
            for name, tensor in hf_named_tensors:
                shard_tensors[name] = tensor.detach().to("cpu").contiguous()
                weight_map[name] = shard_name
                total_size += shard_tensors[name].numel() * shard_tensors[name].element_size()
            try:
                safetensors.torch.save_file(shard_tensors, path / shard_name)
            except Exception as exc:
                # Peers must finish the remaining weight gathers before reporting a write failure.
                write_error = exc
            del shard_tensors

        if write_error is not None:
            raise write_error
        if is_writer:
            assert weight_map, f"HF export to {path} produced no weights"
            base_checkpoint = Path(hf_checkpoint)
            if base_checkpoint.is_dir():
                for meta_file in base_checkpoint.iterdir():
                    if _is_hf_metadata_file(meta_file):
                        shutil.copy2(meta_file, path / meta_file.name)
            else:
                logger.warning(f"hf_checkpoint {hf_checkpoint} is not a local dir; metadata not copied to {path}")
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
