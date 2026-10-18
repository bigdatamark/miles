# [MODULE DOC] Unit tests for the validation of rollout top-k log-probability arguments.
# The tests build a bare argument namespace holding defaults for the rollout top log-prob count,
# the sampling log-prob mode, top-p, top-k and the on-policy-distillation (OPD) settings, apply
# per-case overrides, and call the validator. One parametrized test checks that consistent
# settings pass silently; another checks that inconsistent settings raise a value error whose
# message matches an expected fragment.
from argparse import Namespace

import pytest

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_args


def _args(**overrides: object) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds an argument namespace with valid defaults (zero top log-probs,
    # selected mode, top-p 1.0, top-k -1, OPD disabled with top-k 0 and only-student strategy),
    # then merges the keyword overrides on top (overrides win). Returns the namespace; no side
    # effects.
    values = dict(
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        rollout_top_p=1.0,
        rollout_top_k=-1,
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
    )
    return Namespace(**(values | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"rollout_top_logprobs_num": 128},
        {"rollout_top_logprobs_num": 128, "rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
        {
            "rollout_top_logprobs_num": 64,
            "rollout_sampling_logprobs_mode": "support",
            "rollout_top_p": 0.9,
            "rollout_top_k": 64,
        },
        {"rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
        {"rollout_sampling_logprobs_mode": "support", "rollout_top_p": 0.9, "rollout_top_k": 64},
        {
            "rollout_top_logprobs_num": 8,
            "use_opd": True,
            "opd_log_prob_top_k": 4,
            "opd_top_k_strategy": "only-teacher",
        },
    ],
)
def test_consistent_settings_pass(overrides: dict) -> None:
    # [FUNCTION DOC] Parametrized over seven consistent override sets (defaults; a top log-prob
    # count of 128; support mode with top-k 64 and a sufficient count; the same with top-p 0.9;
    # support mode without a top log-prob count; and an OPD case with a larger rollout count than
    # the OPD top-k under the only-teacher strategy). Asserts the validator returns without
    # raising.
    validate_rollout_topk_logprobs_args(_args(**overrides))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rollout_top_logprobs_num": -1}, "non-negative"),
        (
            {"rollout_top_logprobs_num": 32, "rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
            ">= --rollout-top-k",
        ),
        ({"rollout_top_logprobs_num": 8, "use_opd": True, "opd_log_prob_top_k": 4}, "OPD student"),
    ],
)
def test_inconsistent_settings_are_rejected(overrides: dict, match: str) -> None:
    # [FUNCTION DOC] Parametrized over three invalid override sets: a negative top log-prob count
    # (message mentions non-negative), a count smaller than top-k in support mode (message
    # mentions it must be at least the rollout top-k flag), and OPD with the default only-student
    # strategy where the rollout count exceeds the OPD top-k (message mentions the OPD student).
    # Asserts the validator raises a value error whose message matches the given pattern.
    with pytest.raises(ValueError, match=match):
        validate_rollout_topk_logprobs_args(_args(**overrides))
