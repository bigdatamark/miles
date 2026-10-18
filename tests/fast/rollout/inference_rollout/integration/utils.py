# [MODULE DOC] HELPER: Shared helpers for the integration tests of the inference rollout: an
# expected sample builder for the standard prompt "What is 1+7?" answered by "\boxed{8}",
# sample data rows, the base command-line arguments selecting the modular rollout function, a
# builder of rollout environment configurations, a loader that constructs and calls the rollout
# function in train or eval mode, and a sample filter keeping groups whose first reward is 1.
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
)
from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.types import Sample, WeightVersionsPerCall


def expected_sample(*, group_index: int | None) -> Sample:
    # [FUNCTION DOC] HELPER: Returns the sample a successful standard generation is expected to
    # produce, with the given group index: index 0, prompt "What is 1+7?", the 7 prompt tokens plus
    # 5 response tokens, response text boxed 8 with length 5, label "8", reward 1, no loss mask,
    # one weight version record without spans, the log probability ramp, completed status, empty
    # metadata, zero non-generation time, zeroed speculative info, and prefix cache info with zero
    # cached tokens out of 7 prompt tokens.
    return Sample(
        group_index=group_index,
        index=0,
        prompt="What is 1+7?",
        tokens=[3838, 374, 220, 16, 10, 22, 30, 59, 79075, 90, 23, 92],
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response="\\boxed{8}",
        response_length=5,
        label="8",
        reward=1,
        loss_mask=None,
        weight_versions=[WeightVersionsPerCall(spans=[])],
        rollout_log_probs=[-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125],
        rollout_routed_experts=None,
        remove_sample=False,
        status=Sample.Status.COMPLETED,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=Sample.SpecInfo(
            spec_num_correct_drafts=0, spec_num_proposed_drafts=0, spec_verify_ct=0, completion_tokens=0
        ),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=0, total_prompt_tokens=7),
    )


MODULAR_ROLLOUT_BASE_ARGV = [
    "--rollout-function-path",
    "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
]

MIXED_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "9"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "7"},
]


def integration_env_config(
    extra_argv: list[str],
    data_rows: list[dict] | None = None,
    latency: float = 0.0,
    variant: str = "single_turn",
):
    # [FUNCTION DOC] HELPER: Builds a rollout environment configuration whose extra command-line
    # arguments are, in order, the base arguments choosing the modular inference rollout function,
    # the arguments for the generate variant (default single turn), and the caller's extras. Data
    # rows (default none, meaning the fixture's own) and the mock server latency in seconds
    # (default 0) are passed through.
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant(variant) + extra_argv,
        data_rows=data_rows,
        latency=latency,
    )


def load_and_call_rollout(args, data_source, mode: str = "train") -> RolloutFnOutput:
    # [FUNCTION DOC] HELPER: Loads the rollout function from the path in the arguments (the train
    # function path in train mode, the eval function path otherwise) with a constructor input of
    # the arguments and data source, and calls it once with rollout id 0 using a train input (train
    # mode) or an eval input (any other mode). Returns the rollout output.
    function_path = args.rollout_function_path if mode == "train" else args.eval_function_path
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        function_path,
    )
    if mode == "train":
        return call_rollout_function(fn, RolloutFnTrainInput(rollout_id=0))
    else:
        return call_rollout_function(fn, RolloutFnEvalInput(rollout_id=0))


def load_and_call_train(args, data_source):
    # [FUNCTION DOC] HELPER: Shorthand for loading and calling the rollout in train mode.
    return load_and_call_rollout(args, data_source, mode="train")


def filter_by_reward(args, samples, **kwargs):
    # [FUNCTION DOC] HELPER: Dynamic filter keeping a group whose first sample (or, for nested
    # groups, the first sample of the first inner list) has reward exactly 1; otherwise drops it
    # with the reason "reward_zero". Extra keyword arguments and the arguments are ignored.
    reward = samples[0].reward if not isinstance(samples[0], list) else samples[0][0].reward
    if reward == 1:
        return FilterOutput(keep=True)
    return FilterOutput(keep=False, reason="reward_zero")
