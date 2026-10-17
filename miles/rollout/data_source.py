# [MODULE DOC] Prompt data sources for rollouts. Defines the abstract data source interface (take
# sample groups, return sample groups, save and load state, optional buffer length), a read-only
# source that walks a global tokenized dataset epoch by epoch (optionally reshuffled each epoch,
# with checkpointable position), and a buffered variant that first serves previously returned
# sample groups through a pluggable selection function. The state file path helper is public.
import abc
import copy
import logging
import os
from pathlib import Path

import torch

from miles.utils.data import Dataset
from miles.utils.function_registry import load_function
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def compute_global_dataset_state_path(directory: str, *, rollout_id: int | None) -> str:
    # [FUNCTION DOC] Path of the dataset position checkpoint: "<directory>/rollout/
    # global_dataset_state_dict_<rollout id>.pt". A none rollout id yields the literal text "None" in
    # the file name.
    return os.path.join(directory, f"rollout/global_dataset_state_dict_{rollout_id}.pt")


class DataSource(abc.ABC):
    # [CLASS DOC] Interface of prompt providers. A sample "group" is the list of samples generated from
    # one prompt; the source hands out lists of groups and may accept groups back.

    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        # [FUNCTION DOC] Abstract: return a list of num_samples sample groups.
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        # [FUNCTION DOC] Abstract: accept sample groups back into the source (read-only sources raise).
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        # [FUNCTION DOC] Abstract: persist the source's position/state for the rollout id.
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        # [FUNCTION DOC] Abstract: restore the state saved for the rollout id (none = default).
        """
        Load the state of the data source
        """

    def get_buffer_length(self) -> int | None:
        # [FUNCTION DOC] Optional: number of pending buffered groups, or none for unbuffered sources
        # (the default).
        """Pending-sample backlog, or None for sources without a buffer."""
        return None


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    # [CLASS DOC] Read-only source over the global prompt dataset (or, without a global dataset, blank
    # prompts). State: epoch id, running group index and sample index counters (assigned to emitted
    # samples), the offset into the current epoch's dataset order, and a scratch metadata mapping (TODO
    # in code: remove). Checkpointed by save/load.

    def __init__(self, args):
        # [FUNCTION DOC] Initialize counters to zero. If a global dataset is enabled: load the
        # tokenizer (chat template path, remote code trusted) and the processor from the HF checkpoint;
        # if a details dump directory is set, save the tokenizer there and the processor too when it
        # supports saving; build the dataset from the prompt data with the configured keys, maximum
        # prompt length, chat template options and seed; shuffle it for epoch 0 if shuffling is on.
        # Without a global dataset the dataset is none.
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(
                args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
            )
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                # Bespoke processors (e.g. Inkling's) are not ProcessorMixin and cannot serialise.
                if hasattr(processor, "save_pretrained"):
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # [FUNCTION DOC] Return num_samples sample groups. With a dataset: take that many consecutive
        # prompt samples from the current offset and advance it; if the epoch end is crossed, take the
        # remaining tail, bump the epoch id, reshuffle (if enabled) with the new epoch id as seed, and
        # fill from the head of the new order, setting the offset past what was taken. KNOWN ISSUE: if
        # more samples are requested than the dataset holds, the result is shorter than requested (at
        # most one wrap-around). Without a dataset: blank samples. Then for each prompt sample builds a
        # group of the configured samples per prompt, each a deep copy with the next running group
        # index (shared within the group) and the next running sample index (unique per sample).
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        # [FUNCTION DOC] Always raises a runtime error: this source is read-only.
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        # [FUNCTION DOC] Without a global dataset, does nothing. Otherwise serializes a mapping of
        # offset, epoch id, group index counter, sample index counter and metadata to the state file
        # under the save directory for the rollout id, creating directories first.
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = compute_global_dataset_state_path(self.args.save, rollout_id=rollout_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        # [FUNCTION DOC] Restore position from the state file under the load directory for the rollout
        # id. Returns after a warning (the dataset then starts fresh) if the global dataset is disabled,
        # no load directory is configured, or the file is missing. Otherwise reads it (counters default
        # to zero and metadata to empty for missing keys), and, if shuffling is on, reshuffles the
        # dataset with the restored epoch id. KNOWN ISSUE: the log line "load metadata" prints the
        # metadata held BEFORE the file is read.
        if not self.args.rollout_global_dataset:
            logger.warning("--disable-rollout-global-dataset: the dataset starts where a fresh run's would")
            return

        if self.args.load is None:
            logger.warning("no --load: the dataset starts where a fresh run's would")
            return

        path = compute_global_dataset_state_path(self.args.load, rollout_id=rollout_id)
        if not os.path.exists(path):
            logger.warning(f"no dataset state under {path}: the dataset starts where a fresh run's would")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    # [CLASS DOC] Dataset source plus an in-memory buffer of sample groups returned via add-samples.
    # Requests are served from the buffer first (through the selection function, default: oldest
    # first), the rest from the dataset. The buffer is NOT saved or restored by save/load.

    def __init__(self, args):
        # [FUNCTION DOC] Initialize the base source, an empty buffer, and the buffer selection function:
        # the default pop-first, or the function loaded from the configured path (signature: args,
        # rollout id, buffer list, number of samples; returns the selected groups and removes them from
        # the buffer).
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        # [FUNCTION DOC] Take groups from the buffer first (up to num_samples), then fetch the remaining
        # count from the underlying dataset source and append. Returns the combined list.
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        # [FUNCTION DOC] HELPER: Empty list if the buffer is empty or nothing is requested; otherwise the
        # selection function's result (called with args, no rollout id, the buffer and the count); it
        # is trusted to return at most the count and remove the returned groups from the buffer.
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        # [FUNCTION DOC] Append sample groups to the buffer. Empty input is ignored. Asserts the input
        # is a list of lists and that every group has exactly the configured samples per prompt
        # (assertion errors otherwise; groups before the offending one remain added).
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        # [FUNCTION DOC] Merge the given mapping into the source's metadata mapping (mutation).
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        # [FUNCTION DOC] Return the live metadata mapping (not a copy).
        return self.metadata

    def get_buffer_length(self):
        # [FUNCTION DOC] Number of sample groups currently in the buffer.
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    # [FUNCTION DOC] Default buffer selection: remove and return the oldest groups, at most num_samples
    # of them (fewer if the buffer is smaller), mutating the buffer in place. Args and rollout id are
    # unused.
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
