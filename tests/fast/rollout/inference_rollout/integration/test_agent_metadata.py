# [MODULE DOC] Integration tests that metadata returned by an agent function reaches the custom
# reward model during rollout. A mock tools module global holds the metadata the mock agent
# returns (an agent reward of 42); the custom reward model in this file reads that value from each
# sample's metadata and returns it as the reward. Covered for the agentic tool-call variant with
# the agent run inline and in its own subprocess. Module constants: a two-turn data row, the list
# of agentic variants, and the extra command-line arguments selecting batch size 2, 2 samples per
# prompt and per eval prompt and this file's reward function by import path.
import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample


TWO_TURN_DATA_ROWS = [{"input": [{"role": "user", "content": TwoTurnStub.USER_QUESTION}], "label": "2008"}]

_AGENTIC_VARIANTS = ["agentic_tool_call"]

_METADATA_RM_EXTRA_ARGV = [
    "--rollout-batch-size",
    "2",
    "--n-samples-per-prompt",
    "2",
    "--n-samples-per-eval-prompt",
    "2",
    "--custom-rm-path",
    "tests.fast.rollout.inference_rollout.integration.test_agent_metadata._metadata_reward_function",
]


def _metadata_config_for_variant(variant: str) -> RolloutEnvConfig:
    # [FUNCTION DOC] HELPER: Builds a rollout environment configuration for the variant: the base
    # modular rollout arguments, the variant's arguments and the metadata reward arguments, with
    # the two-turn data rows.
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant(variant) + _METADATA_RM_EXTRA_ARGV,
        data_rows=TWO_TURN_DATA_ROWS,
    )


@pytest.mark.parametrize(
    "variant,rollout_env",
    [pytest.param(v, _metadata_config_for_variant(v), id=v) for v in _AGENTIC_VARIANTS],
    indirect=["rollout_env"],
)
def test_agent_metadata_reaches_reward_model(rollout_env, variant):
    # [FUNCTION DOC] Parametrized over the agentic variants. Sets the mock tools module's global
    # agent return metadata to a mapping with agent reward 42.0 (reset to none in a finally
    # block), makes the mock server follow the two-turn stub, runs a train rollout, and asserts the
    # number of groups equals the rollout batch size and that every sample (flattening nested
    # lists) has reward 42.0, i.e. the reward model saw the agent metadata.
    """Verify that agent metadata is available to --custom-rm-path during reward computation."""
    from miles.utils.test_utils import mock_tools

    mock_tools.AGENTIC_RETURN_METADATA = {"agent_reward": 42.0}
    try:
        env = rollout_env
        env.mock_server.process_fn = TwoTurnStub.process_fn

        out = load_and_call_rollout(env.args, env.data_source, mode="train")

        assert len(out.samples) == env.args.rollout_batch_size
        group = out.samples[0]
        flat = [s for sub in group for s in (sub if isinstance(sub, list) else [sub])]
        for sample in flat:
            assert sample.reward == 42.0, f"RM should have read reward from metadata, got {sample.reward}"
    finally:
        mock_tools.AGENTIC_RETURN_METADATA = None


def _subproc_config() -> RolloutEnvConfig:
    # [FUNCTION DOC] HELPER: Builds a rollout environment configuration whose agent function is
    # the mock agent that returns a reward, run in subprocess mode, together with the base modular
    # rollout, agentic variant and metadata reward arguments and the two-turn data rows.
    argv = extra_argv_for_variant(
        "agentic_tool_call",
        custom_agent_function_path="miles.utils.test_utils.mock_tools.run_agentic_tool_call_returning_reward",
    )
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV
        + argv
        + ["--custom-agent-function-mode", "subproc"]
        + _METADATA_RM_EXTRA_ARGV,
        data_rows=TWO_TURN_DATA_ROWS,
    )


@pytest.mark.parametrize("rollout_env", [pytest.param(_subproc_config(), id="subproc")], indirect=True)
def test_subproc_agent_metadata_reaches_reward_model(rollout_env):
    # [FUNCTION DOC] With the agent running in its own process, makes the mock server follow the
    # two-turn stub, runs a train rollout, and asserts that all flattened samples (there is at
    # least one) have reward 42.0. The 42 comes from the mock agent itself returning that
    # metadata (this test does not set the module global).
    """An agent in its own process still reaches the session server and returns its metadata to the RM."""
    rollout_env.mock_server.process_fn = TwoTurnStub.process_fn

    out = load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train")

    flat = [s for group in out.samples for sub in group for s in (sub if isinstance(sub, list) else [sub])]
    assert flat and all(sample.reward == 42.0 for sample in flat)


async def _metadata_reward_function(args, samples: Sample | list[Sample]) -> float | list[float]:
    # [FUNCTION DOC] HELPER: Custom reward model loaded by import path. For a list of samples
    # returns the list of per-sample metadata rewards, for one sample returns its metadata reward.
    # The arguments are ignored.
    """Custom RM that reads reward from sample.metadata['agent_reward'] — simulates SWE-bench pattern."""
    if isinstance(samples, list):
        return [_extract_metadata_reward(s) for s in samples]
    return _extract_metadata_reward(samples)


def _extract_metadata_reward(sample: Sample) -> float:
    # [FUNCTION DOC] HELPER: Returns the sample metadata's agent reward entry as a float, or 0.0
    # when the entry is missing.
    return float(sample.metadata.get("agent_reward", 0.0))
