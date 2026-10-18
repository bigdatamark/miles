# [MODULE DOC] Integration tests of the dynamic sampling filter in the train rollout. Uses mixed
# correct and incorrect data rows and the shared reward-based filter registered temporarily in the
# function registry under the name "test:filter_by_reward".
from contextlib import nullcontext

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    MIXED_DATA_ROWS,
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.function_registry import function_registry


@pytest.mark.parametrize(
    "rollout_env,use_filter,expect_all_correct",
    [
        pytest.param(
            integration_env_config(["--rollout-batch-size", "4"], data_rows=MIXED_DATA_ROWS),
            False,
            False,
            id="no_filter",
        ),
        pytest.param(
            integration_env_config(
                ["--rollout-batch-size", "3", "--dynamic-sampling-filter-path", "test:filter_by_reward"],
                data_rows=MIXED_DATA_ROWS,
            ),
            True,
            True,
            id="with_filter",
        ),
    ],
    indirect=["rollout_env"],
)
def test_filter_effect(rollout_env, use_filter, expect_all_correct):
    # [FUNCTION DOC] Parametrized: without a filter (batch size 4) incorrect samples must appear;
    # with the reward filter registered (batch size 3) only correct ones may remain. Runs a train
    # rollout (inside the temporary registration when filtering) and checks the set of first-sample
    # rewards of the output groups: exactly {1} when filtering, else it contains 0.
    env = rollout_env
    ctx = function_registry.temporary("test:filter_by_reward", filter_by_reward) if use_filter else nullcontext()

    with ctx:
        out = load_and_call_train(env.args, env.data_source)

    rewards = {group[0].reward for group in out.samples}
    if expect_all_correct:
        assert rewards == {1}, "Filter should keep only correct samples"
    else:
        assert 0 in rewards, "Without filter, incorrect samples should be present"


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            integration_env_config(
                ["--rollout-batch-size", "3", "--dynamic-sampling-filter-path", "test:filter_by_reward"],
                data_rows=MIXED_DATA_ROWS,
            ),
            id="with_filter",
        )
    ],
    indirect=["rollout_env"],
)
def test_unfiltered_raw_reward_still_sees_the_dropped_groups(rollout_env):
    # [FUNCTION DOC] With the reward filter registered and batch size 3, runs a train rollout and
    # asserts the kept groups all have reward 1 while the output metric for the unfiltered raw
    # reward lies strictly between 0 and 1 (so it counts the groups the filter dropped).
    """The accepted-only raw_reward is pinned by the filter; this mean must count what was filtered away."""
    env = rollout_env

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(env.args, env.data_source)

    assert {group[0].reward for group in out.samples} == {1}
    assert 0 < out.metrics["rollout/raw_reward_unfiltered"] < 1
