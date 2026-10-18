# [MODULE DOC] End-to-end test fixture providing a custom generate function (selected through the
# custom generate function path option) that wraps the single-turn generator and may split its one
# sample into two training samples sharing one rollout id, to exercise compacted rollouts.
"""E2E fixture: compact generate fn splitting one rollout execution into two
training samples that share the parent's rollout_id. Use via
--custom-generate-function-path tests.manual.compact_split_generate.generate"""

import copy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # [FUNCTION DOC] Runs the single-turn generator and asserts its output holds exactly one sample
    # object (not a list). Sets that sample's rollout id to its index. If the response has fewer
    # than two tokens or the sample was aborted, returns it alone. Otherwise splits at half the
    # response length (integer division): the first sample is a deep copy truncated to the prompt
    # plus the first half, with response length equal to the cut, the loss mask and rollout log
    # probabilities truncated to the cut when present, and status completed; the second is a deep
    # copy keeping full tokens and response, with the loss mask zeroed for the first half (none if
    # the original had none). Returns both as the output samples. Side effects: mutates the
    # sample returned by the generator and performs whatever generation I/O the single-turn
    # generator does. The second sample keeps the full rollout log probabilities and the original
    # status.
    output = await single_turn_generate(input)
    sample = output.samples
    assert isinstance(sample, Sample)

    sample.rollout_id = sample.index
    if sample.response_length < 2 or sample.status == Sample.Status.ABORTED:
        return GenerateFnOutput(samples=[sample])

    prompt_len = len(sample.tokens) - sample.response_length
    cut = sample.response_length // 2

    first = copy.deepcopy(sample)
    first.tokens = sample.tokens[: prompt_len + cut]
    first.response_length = cut
    first.loss_mask = sample.loss_mask[:cut] if sample.loss_mask is not None else None
    if first.rollout_log_probs is not None:
        first.rollout_log_probs = sample.rollout_log_probs[:cut]
    first.status = Sample.Status.COMPLETED

    second = copy.deepcopy(sample)
    second.loss_mask = [0] * cut + list(sample.loss_mask[cut:]) if sample.loss_mask is not None else None

    return GenerateFnOutput(samples=[first, second])
