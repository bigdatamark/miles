# [MODULE DOC] Integration test of over-sampling in the train rollout: groups are generated in
# rounds of the over-sampling batch size (4) and filtered by the shared reward filter until the
# rollout batch size is filled. Data has only one correct row among four, so the number of rounds
# equals the number of correct groups needed; request counts on the mock server verify rounds.
import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.function_registry import function_registry

_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "wrong"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "wrong"},
]

_BASE_ARGV = [
    "--over-sampling-batch-size",
    "4",
    "--dynamic-sampling-filter-path",
    "test:filter_by_reward",
]


def _over_sampling_config(rollout_batch_size: int):
    # [FUNCTION DOC] HELPER: Builds an integration environment configuration with the given
    # rollout batch size, the over-sampling batch size 4, the registered reward filter, and the
    # four data rows (one correct).
    return integration_env_config(["--rollout-batch-size", str(rollout_batch_size)] + _BASE_ARGV, data_rows=_DATA_ROWS)


@pytest.mark.parametrize(
    "rollout_env,expected_rounds",
    [
        pytest.param(_over_sampling_config(1), 1, id="one_round"),
        pytest.param(_over_sampling_config(2), 2, id="two_rounds"),
    ],
    indirect=["rollout_env"],
)
def test_over_sampling_rounds(rollout_env, expected_rounds):
    # [FUNCTION DOC] Parametrized: batch size 1 expects one round, batch size 2 expects two
    # rounds. Runs a train rollout with the filter registered temporarily and asserts the number
    # of groups equals the batch size, all kept groups have reward 1, and the mock server logged
    # rounds times the over-sampling batch size requests.
    env = rollout_env

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    assert all(group[0].reward == 1 for group in out.samples)

    requests_count = len(env.mock_server.request_log)
    expected_requests = expected_rounds * env.args.over_sampling_batch_size
    assert requests_count == expected_requests, f"Expected {expected_rounds} round(s) = {expected_requests} requests"
