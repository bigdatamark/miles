# [MODULE DOC] Integration tests of train and eval rollouts for multi-turn generation, for the
# engine multi-turn variant and the agentic tool-call variant. The mock inference server follows
# the two-turn stub conversation (a tool call, then a final answer containing "2008"). A custom
# reward function from this file (referenced by import path) rewards a sample 1 when its label text
# occurs in its response. Module constants: a two-turn data row with label "2008", the variant
# names, and extra command-line arguments for batch size 2 and 2 samples per (eval) prompt.
from typing import Any

import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant, listify
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample


TWO_TURN_DATA_ROWS = [{"input": [{"role": "user", "content": TwoTurnStub.USER_QUESTION}], "label": "2008"}]

_VARIANT_NAMES = ["multi_turn", "agentic_tool_call"]

_BASE_EXTRA_ARGV = [
    "--rollout-batch-size",
    "2",
    "--n-samples-per-prompt",
    "2",
    "--n-samples-per-eval-prompt",
    "2",
    "--custom-rm-path",
    "tests.fast.rollout.inference_rollout.integration.test_multi_turn._simple_reward_function",
]


def _config_for_variant(variant: str) -> RolloutEnvConfig:
    # [FUNCTION DOC] HELPER: Builds a rollout environment configuration for the variant from the
    # base modular rollout arguments, the variant's arguments and the shared extra arguments,
    # with the two-turn data rows.
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant(variant) + _BASE_EXTRA_ARGV,
        data_rows=TWO_TURN_DATA_ROWS,
    )


@pytest.mark.parametrize(
    "variant,rollout_env",
    [pytest.param(variant, _config_for_variant(variant), id=variant) for variant in _VARIANT_NAMES],
    indirect=["rollout_env"],
)
@pytest.mark.parametrize("test_type", ["train", "eval"])
def test_rollout(rollout_env, variant, test_type):
    # [FUNCTION DOC] Parametrized over the two variants and over train and eval mode. Makes the
    # mock server follow the two-turn stub and runs the rollout in the mode. In train mode asserts
    # the number of groups equals the batch size and verifies the first group's samples; in eval
    # mode asserts a "toy" dataset entry exists and verifies its samples.
    env = rollout_env
    env.mock_server.process_fn = TwoTurnStub.process_fn

    out = load_and_call_rollout(env.args, env.data_source, mode=test_type)

    if test_type == "train":
        assert len(out.samples) == env.args.rollout_batch_size
        group = out.samples[0]
        _verify_samples(variant, group)
    else:
        assert "toy" in out.data
        samples = out.data["toy"]["samples"]
        _verify_samples(variant, samples)


def _verify_samples(variant: str, samples: list[Any]):
    # [FUNCTION DOC] HELPER: Asserts there are exactly 2 entries (the samples per prompt). Each
    # entry is normalized to a list that must hold exactly one item, which must be a sample; for
    # the agentic variant its metadata must contain the leaf entry attached by the version-2
    # session server. Then runs the single-sample checks on it.
    assert len(samples) == 2, f"n_samples_per_prompt=2, so group should have 2 samples, got {len(samples)}"
    for generated in samples:
        [sample] = listify(generated)
        assert isinstance(sample, Sample), f"{variant} should return Sample trajectories"
        if variant == "agentic_tool_call":
            assert "leaf" in sample.metadata, "session server v2 should attach leaf metadata"
        _verify_sample(sample)


def _verify_sample(sample: Sample, expected_reward: float = 1.0, expect_answer: bool = True):
    # [FUNCTION DOC] HELPER: Asserts the sample is completed, its reward equals the expected
    # reward (default 1.0), and, when an answer is expected (default true), that its response
    # contains "2008".
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == expected_reward, f"Sample should have reward={expected_reward}"
    if expect_answer:
        assert "2008" in sample.response, "Response should contain final answer '2008'"


async def _simple_reward_function(args, samples: Sample | list[Sample]) -> float | list[float]:
    # [FUNCTION DOC] HELPER: Custom reward model loaded by import path. For a list returns the
    # per-sample rewards, for one sample its reward. The arguments are ignored.
    if isinstance(samples, list):
        return [_check_reward(sample) for sample in samples]
    return _check_reward(samples)


def _check_reward(sample: Sample) -> float:
    # [FUNCTION DOC] HELPER: Returns 1.0 if the sample's label text occurs in its response, else
    # 0.0, converted to float.
    # KNOWN ISSUE: when the response is the empty string, the logical-and expression yields the
    # empty string itself and converting it to a float raises a value error instead of giving 0.0
    # (a missing response of none would raise a type error).
    return float(sample.response and (str(sample.label) in sample.response))
