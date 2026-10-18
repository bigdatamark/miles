# [MODULE DOC] HELPER: Test fixture generator that writes a fake training dump directory (rollout,
# evaluation, trajectory, dashboard column and per-rank training data files) by running the real
# production dumping functions on small random samples. Public entry points: the dump generator
# (returns ground truth of per-step shard sample indices and evaluation steps), a function that ages
# file modification times so the reader's visibility guard accepts them, and a function that blanks
# one step's samples as if the step was aborted before any generation. Samples are random but
# reproducible from a seed; every third sample is agentic (multi-turn with mixed weight versions
# and tool messages in a chat prompt). Side effects: writes files, optional tokenizer files, and
# alters file times.
"""Generate a DUMMY ``--dump-details`` directory through miles'
REAL dumping pipeline.

Only the *content* is fake (tiny random rollouts); every file on disk is
produced by the actual production code path, so if a PR changes the dumping
logic — schemas, file naming, DP partition / ``sample_indices`` semantics,
reward post-processing, the batch-global ``raw_reward`` layout — dumps built
here change with it and the reader tests break loudly instead of silently
validating a stale hand-written copy.

Real code invoked per step, in the real call order:

1. ``save_debug_rollout_data``            (rollout + eval dump files)
2. ``convert_samples_to_train_data``      (column dict incl. ``sample_indices``)
3. ``split_train_data_by_dp_raw``         (real seqlen-balanced DP partition)
4. ``process_rollout_data_shard``         (train-side partition pop / reorder)
5. ``save_debug_train_data_for_rank``     (per-rank train dump files)

The only hand-synthesized part is :func:`_apply_dummy_actor_columns`: the
training actor's GPU forward passes cannot run in a CPU test, so the *values*
of its per-token outputs (``log_probs``, ``ref_log_probs``, ``entropy``,
``ref_entropy``, ``advantages``, ``returns``) are random tensors and the
actor-side tensorization of ``tokens``/``loss_masks``/``rollout_log_probs``
is mimicked. Column names there mirror real dumps; the end-to-end guard for
them is the realdata test plus the e2e CI run.
"""

from __future__ import annotations

import os
import random
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import torch

from miles.backends.training_utils.metrics.train_dump import save_debug_train_data_for_rank
from miles.ray.rollout.debug_data import save_dashboard_columns, save_debug_rollout_data
from miles.ray.rollout.train_data_conversion import (
    convert_samples_to_train_data,
    process_rollout_data_shard,
    split_train_data_by_dp_raw,
)
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


@dataclass
class DummyRunTruth:
    # [CLASS DOC] HELPER: Record of what the generator wrote: the number of samples per step, a
    # mapping from step to one list of sample indices per data-parallel shard (as produced by the
    # real partitioning, in dumped row order), and the list of steps that have an evaluation dump.
    """Ground truth returned by :func:`dump_dummy_run` for test assertions."""

    n_samples_per_step: int
    # step -> one list of Sample.index per DP shard, in the dumped row order
    # (taken from the REAL partition output, not recomputed)
    shard_indices: dict[int, list[list[int]]]
    eval_ids: list[int]


def dump_dummy_run(
    dump_dir: Path,
    *,
    steps: int = 2,
    num_prompts: int = 4,
    n_samples_per_prompt: int = 2,
    dp_size: int = 2,
    tp_dup: int = 2,
    max_response_len: int = 16,
    with_entropy: bool = True,
    with_eval: bool = True,
    with_tokenizer: bool = True,
    remove_sample_indices: tuple[int, ...] = (),
    duplicate_first_sample_index: bool = False,
    seed: int = 0,
) -> DummyRunTruth:
    # [FUNCTION DOC] HELPER: Writes a dummy dump into dump_dir. Inputs (all keyword-only after the
    # directory): steps (2), prompts per step (4), samples per prompt (2), data-parallel size (2),
    # tensor-parallel duplication factor (2, each shard saved under that many ranks), maximum
    # response length (16), whether to include entropy columns, an evaluation dump at step zero, and
    # a tokenizer; positions within a step to flag as removed; whether to make the second sample
    # copy the first's index and reward (to test duplicates); and the random seed (0). Asserts the
    # per-step batch size (prompts times samples) is divisible by the data-parallel size. Per step:
    # builds random samples (global index = step times batch size plus position; group index =
    # position integer-divided by samples per prompt; the first two samples get a chat message
    # list in their metadata), saves the rollout dump, optionally saves an evaluation dump with the
    # first half of the samples (step zero only), converts the samples to training data with the real
    # conversion, splits by data parallel rank with the real partition, records the shard indices,
    # then for each shard runs the real shard processing, adds fake actor columns using a
    # generator seeded from the main random stream, and saves it once per tensor-parallel
    # duplicate with rank equal to shard index times the duplication factor plus duplicate index.
    # Finally ages all files and returns the truth record.
    """``remove_sample_indices`` marks the given within-step positions as
    ``remove_sample=True`` in every step; the real conversion then writes an
    all-zero loss mask for them."""
    n = num_prompts * n_samples_per_prompt
    assert n % dp_size == 0, "equal-size DP partition needs divisible batch"
    args = _make_args(dump_dir, num_prompts=num_prompts, n_samples_per_prompt=n_samples_per_prompt)
    rng = random.Random(seed)
    truth = DummyRunTruth(n_samples_per_step=n, shard_indices={}, eval_ids=[])

    if with_tokenizer:
        _write_tokenizer(dump_dir)

    for rollout_id in range(steps):
        samples = [
            _make_sample(
                rng,
                index=rollout_id * n + i,
                group_index=i // n_samples_per_prompt,
                max_response_len=max_response_len,
                remove_sample=i in remove_sample_indices,
            )
            for i in range(n)
        ]
        if duplicate_first_sample_index:
            samples[1].index = samples[0].index
            samples[1].reward = samples[0].reward
        for sample in samples[:2]:
            sample.metadata["messages"] = _dummy_messages(sample)
        save_debug_rollout_data(args, samples, rollout_id=rollout_id, evaluation=False)
        if with_eval and rollout_id == 0:
            # eval takes the real input shape: {dataset_name: {"samples": [...]}}
            eval_data = {"dummy_eval": {"samples": samples[: n // 2]}}
            save_debug_rollout_data(args, eval_data, rollout_id=rollout_id, evaluation=True)
            truth.eval_ids.append(rollout_id)

        train_data = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        shards = split_train_data_by_dp_raw(args, train_data, dp_size=dp_size)
        truth.shard_indices[rollout_id] = [list(shard["sample_indices"]) for shard in shards]
        generator = torch.Generator().manual_seed(rng.randint(0, 2**31))
        for shard_index, shard in enumerate(shards):
            shard = process_rollout_data_shard(args, shard)
            _apply_dummy_actor_columns(shard, generator, with_entropy=with_entropy)
            for dup in range(tp_dup):
                save_debug_train_data_for_rank(
                    args, rollout_id=rollout_id, rollout_data=shard, rank=shard_index * tp_dup + dup
                )

    age_files(dump_dir)
    return truth


def _dummy_messages(sample) -> list[dict]:
    # [FUNCTION DOC] HELPER: Returns a fixed five-message chat transcript: system, user (the
    # sample's prompt as text), an assistant message with content, reasoning content and one tool
    # call to a lookup function, a tool result, and a final assistant reply mentioning the sample's
    # index.
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": str(sample.prompt)},
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "reasoning_content": "The user asks a question; call the lookup tool first.",
            "tool_calls": [{"function": {"name": "lookup", "arguments": '{"query": "answer"}'}}],
        },
        {"role": "tool", "name": "lookup", "content": '{"result": 42}'},
        {"role": "assistant", "content": f"The answer for sample {sample.index} is 42."},
    ]


def _make_args(dump_dir: Path, *, num_prompts: int, n_samples_per_prompt: int) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the argument namespace consumed by the real dump functions:
    # output path templates for rollout data, trajectory data and train data under the dump
    # directory (with rollout id and rank placeholders), data balancing on, dynamic global batch
    # size off, grpo advantage estimator with reward and standard deviation normalization, the given
    # rollout batch size (prompts) and samples per prompt, no reward key, thd qkv format, and zero
    # top log probabilities. Contains only attributes the invoked functions read.
    # Exactly the attributes the invoked dump-pipeline functions read; a new
    # attribute dependency in main code fails loudly here (drift detection).
    return Namespace(
        save_debug_rollout_data=f"{dump_dir}/rollout_data/{{rollout_id}}.pt",
        save_debug_trajectory_data=f"{dump_dir}/trajectory/{{rollout_id}}.jsonl",
        save_debug_train_data=f"{dump_dir}/train_data/{{rollout_id}}_{{rank}}.pt",
        balance_data=True,
        use_dynamic_global_batch_size=False,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        rollout_batch_size=num_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        reward_key=None,
        qkv_format="thd",
        rollout_top_logprobs_num=0,
    )


def _make_sample(
    rng: random.Random, *, index: int, group_index: int, max_response_len: int, remove_sample: bool = False
) -> Sample:
    # [FUNCTION DOC] HELPER: Builds one random sample. Response length is uniform in 4 through the
    # maximum; prompt length uniform in 3 through 8; truncated if the response reached the maximum
    # (status truncated, else completed). Agentic samples (global index divisible by three) have a
    # three-message chat prompt (user, assistant, tool) and two weight-version calls: the first
    # covering the prompt through the response midpoint with version index mod 3, the second the rest
    # with that version plus one; others have a plain prompt string and one version span (index mod 3)
    # covering prompt end to total length. Tokens are random ids below the vocabulary size (100);
    # the response text is "x" repeated response-length times; reward is 1.0 with probability one
    # half, else 0.0; rollout log probabilities are negative random values, one per response token.
    # The remove flag is passed through. Consumes the random generator.
    response_length = rng.randint(4, max_response_len)
    prompt_length = rng.randint(3, 8)
    truncated = response_length == max_response_len
    # every third sample is agentic-shaped: multi-turn (mixed weight versions
    # under fully async) with a chat prompt carrying tool messages
    agentic = index % 3 == 0
    total_length = prompt_length + response_length
    if agentic:
        split = prompt_length + response_length // 2
        versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan(str(index % 3), prompt_length, split)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan(str(index % 3 + 1), split, total_length)]),
        ]
    else:
        versions = [WeightVersionsPerCall(spans=[WeightVersionSpan(str(index % 3), prompt_length, total_length)])]
    prompt: str | list = "What is 1+1?"
    if agentic:
        prompt = [
            dict(role="user", content="What is 1+1?"),
            dict(role="assistant", content="let me check"),
            dict(role="tool", content="2"),
        ]
    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt,
        tokens=[rng.randint(0, _VOCAB_SIZE - 1) for _ in range(prompt_length + response_length)],
        response="x" * response_length,
        response_length=response_length,
        reward=float(rng.random() < 0.5),
        rollout_log_probs=[-rng.random() for _ in range(response_length)],
        weight_versions=versions,
        status=Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED,
        remove_sample=remove_sample,
    )


_VOCAB_SIZE = 100


def _write_tokenizer(dump_dir: Path) -> None:
    # [FUNCTION DOC] HELPER: Builds a word-level tokenizer whose vocabulary maps the text "t<i>" to
    # id i for i below the vocabulary size (unknown token is "t0"), wraps it in a fast pretrained
    # tokenizer and saves it under a tokenizer folder in the dump directory. Imports the tokenizers
    # and transformers libraries lazily; writes files.
    """Persist a tiny word-level tokenizer (token id i <-> text "t{i}") with the
    same ``save_pretrained`` mechanism the real data source uses for
    ``{dump_details}/tokenizer/``."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(_VOCAB_SIZE)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=raw).save_pretrained(dump_dir / "tokenizer")


def _apply_dummy_actor_columns(shard: dict, generator: torch.Generator, *, with_entropy: bool) -> None:
    # [FUNCTION DOC] HELPER: Mutates the shard mapping in place. Converts tokens, loss masks (32-bit
    # integers) and rollout log probabilities lists into tensors per sample; adds per-sample random
    # tensors of each response length (drawn from the given generator, uniform in zero to one) for
    # log probs and reference log probs (negated), advantages and returns, and, if requested,
    # entropy and reference entropy. Per-token values are random, not computed.
    """Mimic what the training actor does to a shard before dumping it:
    tensorize the converted columns and attach forward-pass outputs.
    The values are random; the column names mirror real dumps."""
    response_lengths = shard["response_lengths"]

    def per_token(sign: float = 1.0) -> list[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Returns one random tensor per sample with that sample's response
        # length, multiplied by the sign (default positive one).
        return [sign * torch.rand(r, generator=generator) for r in response_lengths]

    shard["tokens"] = [torch.tensor(t) for t in shard["tokens"]]
    shard["loss_masks"] = [torch.tensor(m, dtype=torch.int32) for m in shard["loss_masks"]]
    shard["rollout_log_probs"] = [torch.tensor(p) for p in shard["rollout_log_probs"]]
    shard["log_probs"] = per_token(sign=-1.0)
    shard["ref_log_probs"] = per_token(sign=-1.0)
    shard["advantages"] = per_token()
    shard["returns"] = per_token()
    if with_entropy:
        shard["entropy"] = per_token()
        shard["ref_entropy"] = per_token()


def age_files(dump_dir: Path, *, seconds: float = 100.0) -> None:
    # [FUNCTION DOC] HELPER: Sets the access and modification times of every tensor dump file found
    # recursively under the directory to the current time minus the given seconds (default 100).
    """Push mtimes into the past so files pass the reader's visibility guard."""
    stamp = time.time() - seconds
    for path in dump_dir.rglob("*.pt"):
        os.utime(path, (stamp, stamp))


def blank_samples(dump_dir: Path, rollout_id: int, *, seconds: float = 100.0) -> None:
    # [FUNCTION DOC] HELPER: Simulates a step aborted before generation. Loads the step's rollout
    # dump (full unpickling on CPU), replaces its samples with an empty list and saves it back;
    # rewrites the step's dashboard columns file with no rows; deletes the step's trajectory file
    # if present and all of the step's per-rank train data files; then sets the rollout dump's file
    # times to now minus seconds (default 100). Raises if the rollout dump does not exist.
    """Rewrite a step as one aborted before any generation landed."""
    path = dump_dir / "rollout_data" / f"{rollout_id}.pt"
    pack = torch.load(path, map_location="cpu", weights_only=False)
    pack["samples"] = []
    torch.save(pack, path)

    save_dashboard_columns([], dump_dir / "dashboard_columns" / f"rollout_{rollout_id}.parquet")
    (dump_dir / "trajectory" / f"{rollout_id}.jsonl").unlink(missing_ok=True)
    for shard in (dump_dir / "train_data").glob(f"{rollout_id}_*.pt"):
        shard.unlink()

    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))
