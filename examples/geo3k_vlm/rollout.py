# [MODULE DOC] Single-turn rollout wrapper for the geo3k vision-language example. Delegates to the
# standard SGLang rollout generate function and then removes non-tensor entries from the sample's
# multimodal training inputs, because the Megatron data path expects every value to be a tensor.
"""Single-turn geo3k VLM rollout that keeps multimodal_train_inputs tensor-only."""

import torch

from miles.rollout.sglang_rollout import generate as _generate
from miles.utils.types import Sample


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    # [FUNCTION DOC] Custom generate function: runs the default generate and then post-processes the
    # result. If the sample has multimodal training inputs, keeps only the entries whose value is a
    # tensor (the processor returns some fields, such as modality type ids, as plain lists which
    # would crash later when moved to a device); if nothing is left sets them to none. Returns the
    # sample (the same object returned by the default generate).
    sample = await _generate(args, sample, sampling_params)
    # Qwen3-VL's processor returns mm_token_type_ids as a Python list (text-modality
    # fields use return_tensors=None). The Megatron data path treats every
    # multimodal_train_inputs value as a torch tensor, so a list-valued field raises
    # "AttributeError: 'list' object has no attribute 'to'". Drop non-tensor fields,
    # matching the multi-turn geo3k example's merge behavior.
    mm = sample.multimodal_train_inputs
    if mm:
        sample.multimodal_train_inputs = {k: v for k, v in mm.items() if isinstance(v, torch.Tensor)} or None
    return sample
