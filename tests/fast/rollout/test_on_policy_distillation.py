# [MODULE DOC] CPU tests for on-policy distillation helpers in the rollout package: the top-k
# reverse KL computation under four strategies (only-student, intersection, only-teacher, xor)
# with hand-computed expectations from synthetic student and teacher log-probability payloads; the
# per-position token id builder and score payload builder; the parser for the multi-teacher URL
# option; and the per-sample teacher URL routing (by metadata name, custom key, default entry,
# fallback to the single remote model URL, and errors). No network or GPU. Log-probability entries
# are pairs of (natural log of a probability, token id).
import math
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.on_policy_distillation import (
    _compute_topk_reverse_kl,
    _per_position_ids,
    _score_payload,
    _teacher_url_for_sample,
    parse_teacher_urls,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


def _entry(prob: float, token_id: int):
    # [FUNCTION DOC] HELPER: Returns a two-element list: the natural log of the probability, then
    # the token id (the wire format of a top-k log-probability entry).
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    # [FUNCTION DOC] HELPER: Builds a namespace with the top-k strategy name and the reward weight
    # mode (default "student_p").
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    # [FUNCTION DOC] HELPER: Builds a sample with tokens 10, 11, 12 and response length 2, whose
    # metadata holds the student's top-2 log-probabilities for the two response positions:
    # position 0 has token 1 at 0.6 and token 2 at 0.4; position 1 has token 4 at 0.7 and token 5
    # at 0.3.
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def _teacher_payload():
    # [FUNCTION DOC] HELPER: Builds a scoring payload with two sections. The teacher section has
    # input top log-probabilities and input token-id log-probabilities for three positions (the first
    # is none, as the prompt-start slot). The student-on-teacher section has the token-id
    # log-probabilities of the student at the teacher's chosen ids. Only the last two positions
    # (the response positions) carry data.
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    # [FUNCTION DOC] With the only-student strategy, the per-position reverse KL equals the sum over
    # the student's top-k tokens of student probability times log of student over teacher
    # probability. Position 0: 0.6*ln(0.6/0.3) + 0.4*ln(0.4/0.7); position 1: 0.7*ln(0.7/0.4) +
    # 0.3*ln(0.3/0.6). Compared approximately.
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    # [FUNCTION DOC] With the intersection strategy and no weighting, only tokens in both the
    # student and teacher top-k contribute: position 0 overlaps on token 2 giving ln(0.4/0.5);
    # position 1 on token 4 giving ln(0.7/0.8).
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    # [FUNCTION DOC] With the only-teacher strategy and a sample lacking student top log-probs, the
    # reverse KL uses the teacher's top-k tokens with the student's probability for them (from the
    # student-on-teacher section), normalized weights: position 0 uses weights 2/3 and 1/3 (0.4 and
    # 0.2 renormalized) times ln(student/teacher); position 1 uses 7/8 and 1/8.
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    # [FUNCTION DOC] With the xor strategy and no weighting, tokens in exactly one of the two top-k
    # sets contribute an unweighted log ratio: position 0 gives ln(0.6/0.3) + ln(0.2/0.5);
    # position 1 gives ln(0.3/0.6) + ln(0.1/0.2).
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_per_position_ids_pads_prompt_and_keeps_response_order():
    # [FUNCTION DOC] Given the student's top-k for two response positions and a prompt length of 3,
    # the per-position id builder returns three empty lists followed by the token id lists of each
    # response position in order: [5, 7] and [9, 11]. Also checks that the last two entries line up
    # with how the response slice of a length-5 sequence is extracted (drop the first, take the last
    # two).
    # Two response positions, each with two top-k entries [logprob, token_id].
    student_top = [[_entry(0.6, 5), _entry(0.4, 7)], [_entry(0.7, 9), _entry(0.3, 11)]]
    per_pos = _per_position_ids(student_top, prompt_len=3)
    # 3 empty prompt slots, then response positions with their own token ids.
    assert per_pos == [[], [], [], [5, 7], [9, 11]]
    # Aligns with the existing _trim_input_field extraction values[1:][-R:]: for a
    # length-5 response, indices 3,4 are the response positions.
    values = list(range(5))
    assert values[1:][-2:] == [3, 4]
    assert per_pos[3] == [5, 7] and per_pos[4] == [9, 11]


def test_score_payload_routes_per_position_vs_flat():
    # [FUNCTION DOC] The score payload builder puts a flat token id list under the token-ids
    # log-prob key (and omits the per-position key), and puts a per-position list of lists under
    # the positions key (omitting the flat key).
    flat = _score_payload([1, 2, 3], token_ids=[5, 7])
    assert flat["token_ids_logprob"] == [5, 7]
    assert "token_ids_logprob_positions" not in flat

    per_pos = _score_payload([1, 2, 3], token_ids_positions=[[], [5, 7], [9, 11]])
    assert per_pos["token_ids_logprob_positions"] == [[], [5, 7], [9, 11]]
    assert "token_ids_logprob" not in per_pos


# ---------------------------------------------------------------------------
# Multi-teacher routing (--opd-teacher-urls)
# ---------------------------------------------------------------------------


def _routing_args(urls=None, key="opd_teacher", rm_url="http://single-teacher/generate"):
    # [FUNCTION DOC] HELPER: Builds a namespace with the teacher URL option list (default none), the
    # metadata key naming a sample's teacher (default "opd_teacher") and the single fallback remote
    # model URL.
    return Namespace(opd_teacher_urls=urls, opd_teacher_key=key, rm_url=rm_url)


def _tagged_sample(metadata=None):
    # [FUNCTION DOC] HELPER: Builds a small sample (tokens 1, 2, 3, response length 2) with the
    # given metadata mapping (empty when none).
    return Sample(tokens=[1, 2, 3], response_length=2, metadata=metadata or {})


def test_parse_teacher_urls_parses_names_and_keeps_equals_in_url():
    # [FUNCTION DOC] NAME=URL entries parse into a mapping from name to URL; only the first equals
    # sign splits, so equals signs inside the URL query are preserved.
    url_map = parse_teacher_urls(["math=http://h1:30001/generate", "code=http://h2:30002/generate?tag=a=b"])
    assert url_map == {
        "math": "http://h1:30001/generate",
        "code": "http://h2:30002/generate?tag=a=b",
    }


def test_parse_teacher_urls_empty_or_none_gives_empty_map():
    # [FUNCTION DOC] A none value and an empty list both parse to an empty mapping.
    assert parse_teacher_urls(None) == {}
    assert parse_teacher_urls([]) == {}


@pytest.mark.parametrize("bad", ["math", "=http://h1/generate", "math=", "  =  "])
def test_parse_teacher_urls_rejects_malformed_entries(bad):
    # [FUNCTION DOC] Parametrized over entries with no equals sign, an empty name, an empty URL, or
    # only whitespace around the equals sign: each raises a value error saying the expected form is
    # NAME=URL.
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_teacher_urls([bad])


def test_parse_teacher_urls_rejects_duplicate_names():
    # [FUNCTION DOC] Two entries with the same name raise a value error mentioning a duplicate
    # teacher name.
    with pytest.raises(ValueError, match="Duplicate teacher name"):
        parse_teacher_urls(["math=http://h1/generate", "math=http://h2/generate"])


def test_routing_unset_map_falls_back_to_rm_url():
    # [FUNCTION DOC] With no teacher URL option set, the teacher URL for any sample (even one tagged
    # with a name) is the single remote model URL.
    args = _routing_args(urls=None)
    sample = _tagged_sample({"opd_teacher": "math"})
    assert _teacher_url_for_sample(args, sample) == "http://single-teacher/generate"


def test_routing_by_metadata_name():
    # [FUNCTION DOC] With two named teachers, a sample whose metadata teacher key is "math" routes to
    # the math URL and one with "code" routes to the code URL.
    args = _routing_args(urls=["math=http://h1/generate", "code=http://h2/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "math"})) == "http://h1/generate"
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "code"})) == "http://h2/generate"


def test_routing_respects_custom_metadata_key():
    # [FUNCTION DOC] When the teacher key option is changed to "task", the name is read from the
    # sample's "task" metadata entry.
    args = _routing_args(urls=["math=http://h1/generate"], key="task")
    assert _teacher_url_for_sample(args, _tagged_sample({"task": "math"})) == "http://h1/generate"


def test_routing_missing_name_uses_default_entry():
    # [FUNCTION DOC] A sample with no teacher name in its metadata routes to the entry named
    # "default" when one exists.
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({})) == "http://h3/generate"


def test_routing_unknown_name_uses_default_entry():
    # [FUNCTION DOC] A sample naming an unlisted teacher ("physics") routes to the "default" entry.
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"})) == "http://h3/generate"


def test_routing_unknown_name_without_default_raises():
    # [FUNCTION DOC] An unlisted teacher name with no "default" entry raises a value error saying
    # the name matches no teacher URL name.
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="matches no --opd-teacher-urls name"):
        _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"}))


def test_routing_missing_name_without_default_raises():
    # [FUNCTION DOC] A sample with no teacher name and no "default" entry raises a value error
    # about a missing teacher key.
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="missing teacher key"):
        _teacher_url_for_sample(args, _tagged_sample({}))
