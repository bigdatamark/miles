# [MODULE DOC] Tests of the reward model hub entry points: the single-sample asynchronous
# reward function and the batched one, driven through the repository's synchronous run helper.
# Arguments are mocks with no custom reward path, reward type or URL unless a test sets them.
# Covered: each built-in reward type, partial credit for F1, random and deterministic random types
# (the latter hashes the response and tokens), reward type from sample metadata, invalid types,
# batched results in order, in-place reward assignment (refusing to overwrite an existing reward),
# empty batches, and mixed types per sample via metadata.
from unittest.mock import MagicMock

import pytest

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.async_utils import run
from miles.utils.types import Sample


@pytest.fixture
def mock_args():
    # [FUNCTION DOC] HELPER: Fixture returning a mock arguments object whose custom reward path,
    # reward type and reward URL are all none.
    args = MagicMock()
    args.custom_rm_path = None
    args.rm_type = None
    args.rm_url = None
    return args


class TestAsyncRm:
    # [CLASS DOC] Tests of the single-sample asynchronous reward function.
    @pytest.mark.parametrize(
        "rm_type,response,label,expected",
        [
            ("math", r"\boxed{42}", "42", 1),
            (
                "math",
                r"<|open|>think<|sep|>work<|close|>think<|open|>response<|sep|>Answer: \boxed{42}<|close|>response",
                "42",
                1,
            ),
            ("math", r"\boxed{wrong}", "42", 0),
            ("f1", "hello world", "hello world", 1.0),
            ("dapo", "Answer: 42", "42", {"score": 1.0}),
            ("deepscaler", r"</think>\boxed{42}", "42", 1),
            ("gpqa", "Answer: A", "A", 1.0),
            ("boxed_f1", r"Final answer is \boxed{hello world}", "hello world", 1.0),
        ],
    )
    def test_rm_types(self, mock_args, rm_type, response, label, expected):
        # [FUNCTION DOC] Parametrized over reward types (math, including a response in a special
        # open/close tagged thinking format, f1, dapo, deepscaler, gpqa, boxed f1). Sets the type
        # on the mock arguments, computes the reward for a sample with the response and label and
        # asserts it equals the expectation; when the expectation is a mapping (dapo) each of its
        # entries must equal the corresponding entry of the reward.
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response=response, label=label)
        reward = run(async_rm(mock_args, sample))
        if isinstance(expected, dict):
            for k, v in expected.items():
                assert reward[k] == v
        else:
            assert reward == expected

    def test_f1_rm_partial(self, mock_args):
        # [FUNCTION DOC] The F1 type with a one-word response against a two-word label gives a
        # reward strictly between 0 and 1.
        mock_args.rm_type = "f1"
        sample = Sample(prompt="", response="hello", label="hello world")
        reward = run(async_rm(mock_args, sample))
        assert 0 < reward < 1

    def test_random_rm(self, mock_args):
        # [FUNCTION DOC] The random type returns 0 or 1.
        mock_args.rm_type = "random"
        sample = Sample(prompt="", response="anything", label="anything")
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_deterministic_random_rm_returns_binary(self, mock_args):
        # [FUNCTION DOC] The deterministic random type, on a sample with tokens, returns 0 or 1.
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello", label="", tokens=[1, 2, 3])
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_deterministic_random_rm_is_deterministic(self, mock_args):
        # [FUNCTION DOC] Computing the deterministic random reward five times for the same sample
        # always yields the same value.
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello world", label="", tokens=[10, 20])
        rewards = [run(async_rm(mock_args, sample)) for _ in range(5)]
        assert len(set(rewards)) == 1

    def test_deterministic_random_rm_differs_by_response(self, mock_args):
        # [FUNCTION DOC] Across 20 samples with different response texts and identical tokens,
        # asserts both reward values 0 and 1 occur (the response influences the value).
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response=f"response_{i}", label="", tokens=[1, 2, 3]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

    def test_deterministic_random_rm_differs_by_tokens(self, mock_args):
        # [FUNCTION DOC] Across 20 samples with the same response text and different token lists,
        # asserts both reward values 0 and 1 occur (the tokens influence the value).
        """Same response with different tokens yields both reward values across many samples."""
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response="same", label="", tokens=[i, i + 1, i + 2]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

    def test_rm_type_from_metadata(self, mock_args):
        # [FUNCTION DOC] With no type in the arguments and the math type in the sample metadata,
        # the reward of a correct boxed answer is 1.
        mock_args.rm_type = None
        sample = Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"})
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    @pytest.mark.parametrize(
        "rm_type,match",
        [
            ("unknown_type", "not implemented"),
            ("", "not specified"),
        ],
    )
    def test_invalid_rm_type_raises(self, mock_args, rm_type, match):
        # [FUNCTION DOC] Parametrized: an unknown type name raises a not-implemented error whose
        # message says not implemented; an empty type raises a not-implemented error whose message
        # says not specified.
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response="test", label="test")
        with pytest.raises(NotImplementedError, match=match):
            run(async_rm(mock_args, sample))


class TestBatchedAsyncRm:
    # [CLASS DOC] Tests of the batched asynchronous reward function.
    @pytest.mark.parametrize(
        "rm_type,samples_data,expected",
        [
            (
                "math",
                [(r"\boxed{42}", "42"), (r"\boxed{100}", "100"), (r"\boxed{wrong}", "42")],
                [1, 1, 0],
            ),
            (
                "f1",
                [("hello world", "hello world"), ("different", "something else")],
                [1.0, 0],
            ),
        ],
    )
    def test_batched_rm(self, mock_args, rm_type, samples_data, expected):
        # [FUNCTION DOC] Parametrized over the math and F1 types. Builds one sample per response
        # and label pair, computes the batched rewards and asserts the returned list equals the
        # expected rewards in input order.
        mock_args.rm_type = rm_type
        samples = [Sample(prompt="", response=r, label=label) for r, label in samples_data]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards == expected

    def test_inplace_set_reward_field(self, mock_args):
        # [FUNCTION DOC] With in-place assignment on, asserts the call returns none and both
        # samples' reward fields are set to 1.
        mock_args.rm_type = "math"
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42"),
            Sample(prompt="", response=r"\boxed{100}", label="100"),
        ]
        result = run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))
        assert result is None
        assert samples[0].reward == 1
        assert samples[1].reward == 1

    def test_inplace_raises_on_existing_reward(self, mock_args):
        # [FUNCTION DOC] A sample that already has a reward of 0.5 makes in-place assignment raise
        # an assertion error mentioning overriding.
        mock_args.rm_type = "math"
        samples = [Sample(prompt="", response=r"\boxed{42}", label="42", reward=0.5)]
        with pytest.raises(AssertionError, match="Overriding"):
            run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))

    def test_empty_samples(self, mock_args):
        # [FUNCTION DOC] An empty sample list gives an empty reward list.
        mock_args.rm_type = "math"
        rewards = run(batched_async_rm(mock_args, []))
        assert rewards == []

    def test_mixed_rm_types_via_metadata(self, mock_args):
        # [FUNCTION DOC] With no type in the arguments, two samples name different types in their
        # metadata (math and F1). Asserts the batched rewards are 1 for the math sample and 1.0
        # for the F1 sample, i.e. each sample is scored by its own type.
        mock_args.rm_type = None
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"}),
            Sample(prompt="", response="hello", label="hello", metadata={"rm_type": "f1"}),
        ]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards[0] == 1
        assert rewards[1] == 1.0
