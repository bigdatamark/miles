# [MODULE DOC]
# Custom generate function that runs the multi-agent pipeline for a sample. A module-level table
# of fixed settings supplies the pipeline function path, the number of parallel solver and
# rewriter workers (5) and the reward weights for incorrect (0.8) and correct (1.2) final answers;
# these are written onto the launch args object on every call and cannot be set from the command
# line. The function returns the pipeline's list of samples in random order.
import random

from miles.utils.function_registry import load_function
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.experimental.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}


async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:
    # [FUNCTION DOC] Generate hook for one sample. Loads the tokenizer for the checkpoint with the
    # configured chat template (trusting remote code) on every call; picks the context limit from the
    # training or evaluation launch arg. Then mutates the shared launch args: stores the sampling
    # parameters, the chosen context limit (overwriting the training limit attribute) and the
    # tokenizer, and sets every entry of the fixed settings table. Loads the pipeline function by
    # path, awaits it with the args and sample, shuffles the returned samples in place (global random
    # state) and returns them.
    # KNOWN ISSUE: during evaluation the training context length attribute on the shared args is
    # overwritten by the evaluation value and never restored, and the fixed settings override any
    # user-provided values; the tokenizer is reloaded per call.

    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    for key, value in MULTI_AGENT_CONFIGS.items():
        setattr(args, key, value)

    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)

    random.shuffle(samples)

    return samples
