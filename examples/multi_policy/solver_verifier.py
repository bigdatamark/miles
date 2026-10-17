# [MODULE DOC] Custom rollout generation and eval logging for the solver/verifier example: for
# every prompt a solver policy answers a math question and a verifier policy then judges that
# answer; both resulting samples are returned and rewarded, and each is stamped with the id of
# the trainer (policy) that should learn from it. Two policies are required, taken in order from
# the multi-policy trainer configuration (first is the solver, second the verifier). Requests go
# to per-policy routers found in the args.
# Rewards: the solver gets 1 if its answer matches the ground truth else 0. The verifier must end
# with exactly one verdict line (agree or wrong); score table: invalid verdict 0; solver correct
# and agreed 1; solver correct but called wrong 0; solver wrong but agreed 0; solver wrong and
# called wrong 1 if the verifier's own final answer is right, else 0.5. Answers are the last
# marked answer after four hash signs, else the last number in the text, with trailing period,
# thousands commas and dollar signs removed, compared as strings.
# See tests/e2e/short/test_multi_policy_solver_verifier_gsm8k.py for an end-to-end run of this example.
import argparse
import dataclasses
import re
from enum import Enum
from typing import Any

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.iter_utils import group_by
from miles.utils.types import Sample

_AGREE_MARKER = "AGREE"
_WRONG_MARKER = "WRONG"
_VERDICT_PREFIX = "VERDICT:"

_VERIFIER_PROMPT_TEMPLATE = (
    "Another model was asked to solve a math problem, and you must check its work.\n\n"
    "Question:\n{question}\n\n"
    "Proposed solution:\n{solver_response}\n\n"
    f"Reason about the proposed solution, then write exactly one verdict line of its own: "
    f"'{_VERDICT_PREFIX} {_AGREE_MARKER}' if the proposed final answer is correct, or "
    f"'{_VERDICT_PREFIX} {_WRONG_MARKER}' if it is not. After a {_WRONG_MARKER} verdict, "
    "write your own final answer on a last line as '#### <answer>'."
)

_VERDICT_PATTERN = re.compile(rf"^{_VERDICT_PREFIX}\s*({_AGREE_MARKER}|{_WRONG_MARKER})\s*$", re.MULTILINE)
_MARKED_ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
_NUMBER_PATTERN = re.compile(r"-?\d+(?:[\d,]*\d)?(?:\.\d+)?")


class _Verdict(Enum):
    # [CLASS DOC] HELPER: The two valid verifier verdicts, valued by their marker words (agree
    # means the solver's answer is judged correct, wrong means judged incorrect).
    AGREE = _AGREE_MARKER
    WRONG = _WRONG_MARKER


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # [FUNCTION DOC] Custom generate function (registered through the custom generate path flag).
    # Input: the standard generate input whose sample carries the question prompt (must be a
    # chat message list) and label. Steps: resolve the policy ids from the megatron config of the
    # args and assert there are exactly two (first solver, second verifier); run the standard
    # single-turn generation for the solver at its router and require a single sample that is not
    # aborted; build a verifier sample whose prompt embeds the question and the solver's response;
    # run single-turn generation for it at the verifier router (the input is copied with the
    # sample replaced) and require a single sample. Extract the ground truth from the label,
    # decide solver correctness, set the solver reward to a mapping with a numeric value (1 or 0)
    # and an outcome name; compute the verifier score from solver correctness, parsed verdict and
    # verifier correctness, set its reward likewise; stamp each sample with its policy id.
    # Returns an output holding the list [solver sample, verifier sample].
    args = input.args
    model_ids = resolve_megatron_config(args).model_ids
    assert len(model_ids) == 2, (
        f"examples/multi_policy/solver_verifier.py pairs one solver policy with one verifier policy, but "
        f"--megatron-config names {model_ids}"
    )
    solver_model_id, verifier_model_id = model_ids

    solver_output = await single_turn_generate(input, url=_compute_router_url(args, model_id=solver_model_id))
    solver_sample = solver_output.samples
    assert isinstance(solver_sample, Sample), f"{solver_sample=}"
    assert solver_sample.status != Sample.Status.ABORTED

    verifier_sample = _build_verifier_sample(solver_sample)
    verifier_output = await single_turn_generate(
        dataclasses.replace(input, sample=verifier_sample),
        url=_compute_router_url(args, model_id=verifier_model_id),
    )
    verifier_sample = verifier_output.samples
    assert isinstance(verifier_sample, Sample), f"{verifier_sample=}"

    ground_truth = _extract_answer(solver_sample.label or "")
    solver_correct = _is_correct(solver_sample.response, ground_truth=ground_truth)
    solver_sample.reward = {
        "reward_value": 1.0 if solver_correct else 0.0,
        "outcome": "solver_correct" if solver_correct else "solver_wrong",
    }
    verifier_reward, verifier_outcome = _compute_verifier_score(
        solver_correct=solver_correct,
        verdict=_parse_verdict(verifier_sample.response),
        verifier_correct=_is_correct(verifier_sample.response, ground_truth=ground_truth),
    )
    verifier_sample.reward = {
        "reward_value": verifier_reward,
        "outcome": verifier_outcome,
    }

    solver_sample.trainer_model_id = solver_model_id
    verifier_sample.trainer_model_id = verifier_model_id
    return GenerateFnOutput(samples=[solver_sample, verifier_sample])


def split_eval_data_by_policy(
    rollout_id: int, args: argparse.Namespace, data: dict[str, dict[str, Any]], extra_metrics: dict[str, Any] | None
) -> bool:
    # [FUNCTION DOC] Eval logging hook (registered through the custom eval log function flag).
    # Rewrites the mapping of eval results in place: each dataset entry is removed and replaced by
    # one entry per policy named dataset name, slash, policy id, containing that policy's samples,
    # their rewards (taken with the eval reward key, else the main reward key, from each sample's
    # reward mapping; none for a sample without reward) and a truncated flag per sample. Asserts
    # every sample carries a policy id. The rollout id and extra metrics inputs are unused.
    # Always returns false.
    reward_key = args.eval_reward_key or args.reward_key
    for name in list(data):
        entry = data.pop(name)
        for model_id, samples in group_by(entry["samples"], lambda sample: sample.trainer_model_id).items():
            assert model_id is not None, (
                f"an eval sample of dataset {name!r} carries no trainer_model_id, so its reward cannot be "
                f"attributed to a policy; the generate function must stamp every sample it returns"
            )
            data[f"{name}/{model_id}"] = dict(
                rewards=[sample.reward[reward_key] if sample.reward is not None else None for sample in samples],
                truncated=[sample.status == Sample.Status.TRUNCATED for sample in samples],
                samples=samples,
            )
    return False


def _compute_verifier_score(
    *, solver_correct: bool, verdict: _Verdict | None, verifier_correct: bool
) -> tuple[float, str]:
    # [FUNCTION DOC] HELPER: Verifier reward table, keyword-only inputs. Returns a pair of reward
    # and outcome name: no valid verdict gives 0 (invalid verdict); with a correct solver, agreeing
    # gives 1 (right solver agreed) and calling it wrong gives 0 (right solver called wrong); with
    # a wrong solver, agreeing gives 0 (wrong solver agreed), and calling it wrong gives 1 if the
    # verifier's own answer is correct (caught and fixed) else 0.5 (caught, not fixed).
    if verdict is None:
        return 0.0, "invalid_verdict"
    if solver_correct:
        return (1.0, "right_solver_agreed") if verdict is _Verdict.AGREE else (0.0, "right_solver_called_wrong")
    if verdict is _Verdict.AGREE:
        return 0.0, "wrong_solver_agreed"
    return (1.0, "wrong_solver_caught_and_fixed") if verifier_correct else (0.5, "wrong_solver_caught_not_fixed")


def _parse_verdict(response: str) -> _Verdict | None:
    # [FUNCTION DOC] HELPER: Find verdict lines (a whole line consisting of the verdict prefix and
    # one marker word, optional spaces after the colon and at the end). Returns the verdict only
    # if exactly one such line exists; zero or several lines give none.
    if len(found := _VERDICT_PATTERN.findall(response)) != 1:
        return None
    return _Verdict(found[0])


def _compute_router_url(args, *, model_id: str) -> str:
    # [FUNCTION DOC] HELPER: Look up the policy's router host and port in the per-policy router
    # mapping of args (a key error for an unknown policy) and return the http generate endpoint
    # URL as a string.
    host, port = args.sglang_model_routers[model_id]
    return f"http://{host}:{port}/generate"


def _build_verifier_sample(solver_sample: Sample) -> Sample:
    # [FUNCTION DOC] HELPER: Build a fresh sample for the verifier. Its prompt is a single user
    # message from the verifier template filled with the solver sample's question and response.
    # Group index, index, rollout id, label and routing key are copied; metadata is a shallow
    # copy of the solver's (or empty). Nothing else (tokens, reward) is carried over.
    prompt = _VERIFIER_PROMPT_TEMPLATE.format(
        question=_extract_question(solver_sample.prompt), solver_response=solver_sample.response
    )
    return Sample(
        group_index=solver_sample.group_index,
        index=solver_sample.index,
        rollout_id=solver_sample.rollout_id,
        prompt=[dict(role="user", content=prompt)],
        label=solver_sample.label,
        metadata=dict(solver_sample.metadata or {}),
        routing_key=solver_sample.routing_key,
    )


def _extract_question(prompt: str | list[dict[str, str]]) -> str:
    # [FUNCTION DOC] HELPER: Return the content of the single user message in a chat message
    # prompt. Asserts the prompt is not a raw string (it could already contain template special
    # tokens). Unpacking fails with an error unless exactly one user message exists.
    assert not isinstance(prompt, str), (
        "examples/multi_policy/solver_verifier.py quotes the question inside the verifier prompt, and a raw "
        "string prompt may already be chat templated with special tokens, so the dataset must use messages"
    )
    [user_content] = [message["content"] for message in prompt if message["role"] == "user"]
    return user_content


def _is_correct(response: str, *, ground_truth: str | None) -> bool:
    # [FUNCTION DOC] HELPER: True only if a ground truth exists and the answer extracted from the
    # response equals it exactly as normalized strings; a missing ground truth is always false.
    if ground_truth is None:
        return False
    return _extract_answer(response) == ground_truth


def _extract_answer(text: str) -> str | None:
    # [FUNCTION DOC] HELPER: Extract the final answer from text. If any four-hash marker lines
    # exist, use the text after the last marker up to the end of its line; otherwise use the last
    # number found (optional minus, digits with optional comma groups, optional decimal part).
    # The chosen string is normalized. Returns none when neither is found.
    if found := _MARKED_ANSWER_PATTERN.findall(text):
        return _normalize_answer(found[-1])
    if found := _NUMBER_PATTERN.findall(text):
        return _normalize_answer(found[-1])
    return None


def _normalize_answer(answer: str) -> str:
    # [FUNCTION DOC] HELPER: Strip surrounding whitespace, remove trailing periods, delete all
    # commas and dollar signs, and strip whitespace again.
    return answer.strip().rstrip(".").replace(",", "").replace("$", "").strip()
