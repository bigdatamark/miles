# [MODULE DOC] Table-driven tests of the DeepScaler rule-based reward function, which returns 1
# or 0 by comparing the boxed answer in the final part of a response (the text after the last
# thinking-end marker, or after a response header marker) with the label.
import pytest

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


class TestGetDeepscalerRuleBasedReward:
    # [CLASS DOC] Tests of the reward function over a table of response and label cases.
    @pytest.mark.parametrize(
        "response,label,expected",
        [
            (r"Let me analyze...</think>The answer is \boxed{42}", "42", 1),
            (r"Thinking...</think>The answer is \boxed{wrong}", "42", 0),
            (r"###Response\boxed{42}", "42", 1),
            (r"###Response\boxed{wrong}", "42", 0),
            (r"The answer is \boxed{42}", "42", 0),
            (r"</think>The answer is 42", "42", 0),
            (r"</think>\boxed{42}", "", 0),
            (r"</think>\boxed{42}", r"\boxed{42}", 1),
            (r"</think>\boxed{123}", 123, 1),
            (r"</think>\boxed{3.14}", 3.14, 1),
            (r"</think>\boxed{1/2}", "0.5", 1),
            (r"</think>\boxed{\frac{1}{2}}", "0.5", 1),
            (r"First thought</think>Second thought</think>\boxed{42}", "42", 1),
        ],
    )
    def test_get_deepscaler_rule_based_reward(self, response, label, expected):
        # [FUNCTION DOC] Asserts the reward equals the expected 1 or 0. Cases: a correct or wrong
        # boxed answer after a thinking-end marker or a response header marker; no marker at all
        # gives 0; a marker without a boxed answer gives 0; an empty label gives 0; labels given
        # as a boxed expression, an integer, a float, a decimal string versus fraction or LaTeX
        # fraction answers are matched mathematically; with several thinking-end markers the
        # last one splits the response.
        assert get_deepscaler_rule_based_reward(response, label) == expected
