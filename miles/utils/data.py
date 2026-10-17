# [MODULE DOC] Prompt dataset loading and rollout-data helpers.
# Provides: a file reader for jsonl / parquet prompt files (with an optional row-slice suffix on
# the path), prompt length filtering, conversion of raw rows to chat messages (including
# multimodal placeholder expansion), the Dataset class (builds Sample objects, supports
# per-epoch deterministic shuffling), a first-fit micro-batch counter, helpers that fetch and
# split a rollout data shard from the object store for one data-parallel rank, and helpers that
# delete stored rollout/train-output objects. Only Dataset is exported by the all list.
import itertools
import json
import logging
import os
import random
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw
from miles.utils import object_store
from miles.utils.pydantic_utils import StrictBaseModel
from .audit_utils.witness.allocator import WitnessInfo

if TYPE_CHECKING:
    from miles.backends.training_utils.types import TrainStepOutput

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from miles.utils import chat_template_utils
from miles.utils.types import MultimodalTypes, Sample

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    # [FUNCTION DOC] Generator yielding one record (a mapping) at a time from a prompt file.
    # Input: a path, optionally suffixed with a row range in square brackets (see the path parser).
    # Raises a file-not-found error if the file is missing, an import error if a parquet file is
    # requested without the parquet library, a value error for any extension other than jsonl or
    # parquet. Jsonl: blank lines skipped; lines that fail to parse are reported with a printed
    # message and skipped. Parquet: reads record batches and yields rows as mappings. If a row
    # range was given, only that range of the yielded rows is produced (logged at info level).
    # Errors for missing file / bad extension surface on first iteration since this is a generator.
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            # [FUNCTION DOC] HELPER: Generator over a utf-8 text file; yields the parsed json value
            # of each non-blank line, printing the zero-based line number and error for undecodable
            # lines and continuing.
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            # [FUNCTION DOC] HELPER: Generator opening the parquet file and yielding every row of
            # every record batch as a mapping from column name to value, in file order.
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    # [FUNCTION DOC] HELPER: Splits an optional trailing row range of the form "@[start:end]" off
    # a path string, where start and end are optional integers (negative allowed by the pattern).
    # Returns (path without the suffix, slice object with missing bounds as none), or (the string
    # unchanged, none) when there is no suffix.
    # KNOWN ISSUE: negative bounds are accepted here but the later iterator slicing rejects
    # negative values at read time.
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    # [FUNCTION DOC] Drops samples whose prompt token count exceeds max_length. Returns the input
    # list unchanged when max_length is none, and also (with a warning) when the first sample's
    # prompt is not a string (list prompts are not length-filtered); an empty input list with a
    # non-none max_length raises an index error. Without a processor: tokenizes all prompts in one
    # batch without special tokens. With a processor: samples with any non-none multimodal input
    # value go through the vision-info helper and the processor (first row's input ids counted),
    # text-only samples are batch-tokenized. Logs the number removed.
    # KNOWN ISSUE: with a processor the result lists text-only survivors first and multimodal
    # survivors after, so the original sample order is not preserved.
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # Use processor only for samples with actual multimodal content; use batched tokenizer for text-only.
        text_only = []
        multimodal = []
        for sample in origin_samples:
            if sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
                multimodal.append(sample)
            else:
                text_only.append(sample)
        filtered_samples = []
        if text_only:
            prompts = [s.prompt for s in text_only]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            for sample, input_ids in zip(text_only, input_ids_list, strict=True):
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
        if multimodal:
            from miles.utils.processing_utils import process_vision_info

            for sample in multimodal:
                multimodal_inputs = process_vision_info(sample.prompt, processor)
                processor_output = processor(text=sample.prompt, **multimodal_inputs)
                input_ids = processor_output["input_ids"][0]
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    # [FUNCTION DOC] HELPER: Turns one raw row into the prompt value for a sample. A string prompt
    # is returned as is unless as_conversation is set, in which case it becomes a single user
    # message. If multimodal_keys (mapping from modality name to the row column holding the media
    # list) is given: for each known modality builds a lookup from its placeholder text to a copy
    # of the media list; then for every message whose content is a string, splits the content on
    # the placeholders and rebuilds it as a list of parts, a text part per plain segment and a
    # media part per placeholder consuming media items in order from the front. Messages whose
    # content is already a list are left untouched with a warning; any other content type raises a
    # value error. Mutates the message mappings inside the row's prompt in place. Returns the
    # prompt (string or list of messages). Unknown modality names are skipped; a missing media
    # column or too few media items for the placeholders causes a type or index error.
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return prompt


class Dataset:
    # [CLASS DOC] In-memory prompt dataset of Sample objects. Attributes: origin_samples (filtered
    # samples in file order), samples (current ordering, initially the same list object), epoch_id
    # (last shuffled epoch, starts at -1) and seed. Supports indexing and length. Shuffling is
    # deterministic per epoch.
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        # [FUNCTION DOC] Reads every row of the file at the path and builds one Sample per row.
        # Keyword options: prompt_key column holding the prompt (default "text"); multimodal_keys
        # mapping modality to media column; label_key column for the label (none means no label;
        # a missing column raises a key error); tool_key column holding tool definitions (a json
        # string is parsed, an array converted to a list, must end up a list or an assertion
        # fails; stored into the sample metadata under tools); metadata_key column of metadata
        # (default "metadata", missing or empty gives an empty mapping); seed (default 42);
        # apply_chat_template flag with extra template kwargs. Rows are treated as conversations
        # when the chat template is applied or multimodal keys exist. With the chat template
        # applied, the prompt becomes the rendered text including the generation prompt (tools
        # passed). If a processor is given and the prompt is a message list, multimodal inputs are
        # extracted via the vision-info helper. After building, samples longer than max_length
        # tokens are filtered out (skipped when max_length is none). Sets epoch id -1 and the
        # working sample list to the origin list.
        origin_samples = []
        for data in read_file(path):
            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            as_conversation = apply_chat_template or (multimodal_keys is not None)
            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            tools = None
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                output_prompt = chat_template_utils.apply_chat_template(
                    prompt,
                    tokenizer=tokenizer,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                output_prompt = prompt

            if processor and isinstance(prompt, list):
                from miles.utils.processing_utils import process_vision_info

                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        # [FUNCTION DOC] Reorders the working samples for an epoch. Does nothing if that epoch is
        # already current. Otherwise seeds the global random generator with seed plus the epoch id
        # (side effect on global random state), shuffles a permutation of indices, rebuilds the
        # working list from the original (unshuffled) samples in that order, and records the
        # epoch id. The same epoch id always yields the same order.
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        # [FUNCTION DOC] Returns the sample at the index in the current (possibly shuffled) order.
        return self.samples[idx]

    def __len__(self):
        # [FUNCTION DOC] Number of samples in the current ordering.
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # [FUNCTION DOC] Returns the number of micro batches a first-fit packing needs. Walks the
    # sequence lengths in the given order and puts each into the first existing batch whose token
    # sum would stay within the per-GPU token limit, else opens a new batch. This is a greedy
    # estimate, not guaranteed minimal; an item longer than the limit gets a batch of its own.
    # Empty input returns zero.
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(
    args,
    rollout_data_ref,
    dp_rank,
    dp_size,
    witness_info: WitnessInfo | None,
) -> tuple[dict, object_store.ObjectStoreGetResult]:
    # [FUNCTION DOC] Fetches this data-parallel rank's rollout data from the object store and
    # returns (processed shard, the store get result). Two modes by the delay-split option in args:
    # (1) delayed split: the reference is a single object; fetch it, add the witness ids under the
    # sequence witness key when witness info is given, split the whole data by data-parallel size
    # with the shared splitter and take the shard for this rank. (2) pre-split: asserts the
    # reference list has one entry per rank and that no witness info is given, fetches this rank's
    # entry and copies it into a plain mapping. Finally runs the shared shard-processing function
    # (imported locally, probably to avoid an import cycle) on the shard.
    from miles.ray.rollout.train_data_conversion import process_rollout_data_shard

    store = object_store.get_instance()

    if args.delay_split_train_data_by_dp:
        get_result = store.get(rollout_data_ref)
        raw = get_result.value
        if (x := witness_info) is not None:
            raw = {**raw, "seq_witness_ids": x.witness_ids}
        raw = split_train_data_by_dp_raw(args, raw, dp_size=dp_size)
        rollout_data = raw[dp_rank]
    else:
        assert len(rollout_data_ref) == dp_size
        assert witness_info is None
        get_result = store.get(rollout_data_ref[dp_rank])
        rollout_data = dict(get_result.value)

    return process_rollout_data_shard(args, rollout_data), get_result


class RolloutDataPack(StrictBaseModel):
    # [CLASS DOC] Strict data model describing stored rollout data: optional list of sample
    # indices, and an optional object-store reference that is either one reference or a list of
    # them (default none for both).
    sample_indices: list[int] | None = None
    data_ref: object_store.StoreObjectRef | list[object_store.StoreObjectRef] | None = None

    @property
    def data_refs(self) -> list[object_store.StoreObjectRef]:
        # [FUNCTION DOC] Property normalizing the reference field to a list: empty list when none,
        # the list itself when already a list, otherwise a one-element list.
        if (ref := self.data_ref) is None:
            return []
        return ref if isinstance(ref, list) else [ref]


def remove_rollout_data_refs(args, rollout_data_pack: RolloutDataPack) -> None:
    # [FUNCTION DOC] Deletes every object referenced by the pack from the object store (side
    # effect on the shared store).
    # KNOWN ISSUE: the args parameter is unused.
    store = object_store.get_instance()
    for ref in rollout_data_pack.data_refs:
        store.remove(ref)


def remove_train_output_refs(train_outputs: Sequence["TrainStepOutput"]) -> None:
    # [FUNCTION DOC] For each train step output that has a values reference, deletes that object
    # from the object store; outputs without one are skipped.
    store = object_store.get_instance()
    for train_output in train_outputs:
        if (ref := train_output.values) is not None:
            store.remove(ref)
