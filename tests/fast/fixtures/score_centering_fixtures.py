# [MODULE DOC]
# Helpers for score-centering integration tests: a tiny tokenizer stand-in, a full argument namespace
# with score-centering defaults, a builder of one fully populated completed sample with rollout top
# logprobs, and a builder of a mock engine meta info mapping. All names are module private.
"""Small deterministic samples for score-centering integration tests."""

from argparse import Namespace

import numpy as np

from miles.rollout.generate_utils.rollout_topk_logprobs import append_rollout_topk_logprobs
from miles.utils.types import Sample


class _Tokenizer:
    # [CLASS DOC] HELPER: Minimal tokenizer stand-in that only supports decoding.
    def decode(self, tokens: list[int]) -> str:
        # [FUNCTION DOC] HELPER: Returns the text form of the token id list (its printed list form),
        # not a real detokenization.
        return str(tokens)


def _args(**overrides: object) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns a namespace of defaults for a score-centering loss run (loss type
    # score centering, three rollout top logprobs in selected mode, no opd, importance sampling none with
    # truncated clip 2.0 and mismatch bounds 0.5 to 5.0, rollout temperature 0.7, top-p 1, top-k -1,
    # grpo advantage without normalization, thd qkv format, log-prob chunk size 2, vocab size 8, zero
    # entropy coefficient, no kl loss, and other flags off). Keyword overrides replace or add entries.
    values = dict(
        ci_test=False,
        loss_type="score_centering",
        rollout_top_logprobs_num=3,
        rollout_sampling_logprobs_mode="selected",
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        rollout_temperature=0.7,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        use_sampling_support_replay=False,
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        use_dynamic_global_batch_size=False,
        balance_data=False,
        qkv_format="thd",
        true_on_policy_mode=False,
        allgather_cp=False,
        log_probs_chunk_size=2,
        vocab_size=8,
        entropy_coef=0,
        observe_training_entropy=False,
        use_kl_loss=False,
    )
    return Namespace(**(values | overrides))


def _turn(prompt: list[int], output: list[int], probabilities: list[float]) -> Sample:
    # [FUNCTION DOC] HELPER: Builds one completed sample (index 0, group 0, reward 1.0). Tokens are the
    # prompt followed by the output; response length is the output length; response text is the printed
    # output list; rollout log probs are the natural log of the given per-token probabilities (so the
    # lists must have equal length, else an error is raised by the strict zip); loss mask is all ones
    # over the output. Then builds an engine-style meta mapping where every output position lists the
    # same full row of (logprob, token, none) entries as its top logprobs, and appends top logprobs for
    # three ranks to the sample via the rollout top-logprobs utility (mutating the sample). Returns it.
    logps = np.log(probabilities).tolist()
    sample = Sample(
        tokens=prompt + output,
        response_length=len(output),
        response=str(output),
        rollout_log_probs=logps,
        loss_mask=[1] * len(output),
        status=Sample.Status.COMPLETED,
        index=0,
        group_index=0,
        reward=1.0,
    )
    rows = [[(logp, token, None) for logp, token in zip(logps, output, strict=True)]] * len(output)
    append_rollout_topk_logprobs(
        sample,
        {
            "output_token_logprobs": [(logp, token, None) for logp, token in zip(logps, output, strict=True)],
            "output_top_logprobs": rows,
        },
        3,
    )
    return sample


def _meta(output: list[int], probabilities: list[float]) -> dict:
    # [FUNCTION DOC] HELPER: Builds an engine meta info mapping: output token logprobs as a list of
    # (log probability, token, none) triples; output top logprobs repeating that whole list once per
    # output token; and a finish reason of type stop. Lengths must match (strict pairing raises).
    entries = [(float(np.log(p)), token, None) for p, token in zip(probabilities, output, strict=True)]
    return {
        "output_token_logprobs": entries,
        "output_top_logprobs": [entries] * len(output),
        "finish_reason": {"type": "stop"},
    }
