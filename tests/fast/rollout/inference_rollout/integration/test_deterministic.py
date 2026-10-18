# [MODULE DOC] Integration test of per-request sampling seeds: with deterministic inference
# enabled and a base rollout seed, each sample of a group gets a consecutive seed; without it no
# seed is sent. Checked on the request log of the mock inference server.
import pytest

from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config, load_and_call_train


@pytest.mark.parametrize(
    "rollout_env,expected_seeds",
    [
        pytest.param(
            integration_env_config(
                [
                    "--sglang-enable-deterministic-inference",
                    "--rollout-seed",
                    "42",
                    "--n-samples-per-prompt",
                    "3",
                    "--rollout-batch-size",
                    "1",
                ]
            ),
            {42, 43, 44},
            id="enabled",
        ),
        pytest.param(
            integration_env_config(["--n-samples-per-prompt", "2", "--rollout-batch-size", "1"]),
            {None},
            id="disabled",
        ),
    ],
    indirect=["rollout_env"],
)
def test_sampling_seeds(rollout_env, expected_seeds):
    # [FUNCTION DOC] Parametrized: "enabled" (deterministic inference, seed 42, 3 samples per
    # prompt, batch size 1) expects the set of sampling seeds {42, 43, 44}; "disabled" (2 samples
    # per prompt) expects only the absent seed (none). Runs a train rollout and asserts the set of
    # sampling seeds found in the mock server's request log equals the expectation.
    env = rollout_env
    load_and_call_train(env.args, env.data_source)

    seeds = {req.get("sampling_params", {}).get("sampling_seed") for req in env.mock_server.request_log}
    assert seeds == expected_seeds
