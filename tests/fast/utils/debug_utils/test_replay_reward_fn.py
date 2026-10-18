# [MODULE DOC] Unit test for the debug reward-replay tool: checks that its asynchronous main
# routine can resolve a custom reward function that exists only in the function registry (not as
# an importable module path) and prints the resulting reward. Uses a temporary registry entry and
# captured stdout; no network or GPU.
from miles.utils.debug_utils.replay_reward_fn import _main_async
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


class TestMainAsync:
    # [CLASS DOC] Tests of the asynchronous main routine of the reward replay tool, which scores a
    # list of samples with the reward function named by a custom reward model path.
    async def test_main_async_resolves_a_registry_only_reward_function(self, capsys):
        # [FUNCTION DOC] Async test. Registers, temporarily and under the name "test:replay_rm", an
        # async reward function that records each sample it receives and returns 0.25; runs the
        # main routine with one sample (index 0, prompt and response strings) and that name.
        # Asserts the recorded samples equal the input list and that standard output contains the
        # text "Reward:   0.25" (with three spaces after the colon). The registry entry is removed
        # on exit of the context.
        """A registry-only name is not an importable module path, so replay must resolve it via function_registry."""
        scored = []

        async def reward_fn(args, sample):
            # [FUNCTION DOC] HELPER: Fake reward function; appends the sample to the recorded
            # list in the enclosing test, ignores the args, and returns the constant reward 0.25.
            scored.append(sample)
            return 0.25

        samples = [Sample(index=0, prompt="prompt", response="response")]

        with function_registry.temporary("test:replay_rm", reward_fn):
            await _main_async(samples=samples, custom_rm_path="test:replay_rm")

        assert scored == samples
        assert "Reward:   0.25" in capsys.readouterr().out
