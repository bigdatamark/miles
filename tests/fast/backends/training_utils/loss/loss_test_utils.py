# [MODULE DOC] HELPER: Shared utilities for loss function snapshot tests. Provides a trivial
# single-process parallel state, default argument sets for the loss code paths, deterministic
# seeded input generation (per-sample tensors plus packed or padded logits), builders for the
# rollout-data and batch dictionaries consumed by advantage and loss functions, deep cloning, a
# recursive output comparison with an absolute tolerance on tensors, and snapshot save/load plus
# lookup of a snapshot directory (shallow-cloning an external artifacts repository into a cache
# under the home directory when no local snapshot files exist).
# KNOWN ISSUE: the usage example in the docstring below mentions a function that runs all loss
# functions, which is not defined in this module (it lives in the callers).
"""Utilities for loss function snapshot testing.

Provides deterministic input generation, snapshot save/load, and recursive
tensor comparison. Designed to be reusable across refactors — any code path
that implements the same function signatures can be validated against snapshots.

Usage:
    # Generate inputs for a config
    parallel_state = make_parallel_state()
    args = make_args(advantage_estimator="grpo", loss_type="policy_loss")
    inputs = make_inputs(seed=42, batch_size=3, prompt_lens=[20, 64, 40],
                         response_lens=[10, 48, 32], vocab_size=128, args=args)

    # Run functions, collect outputs
    outputs = run_all_functions(args, parallel_state, inputs)

    # Save / load
    save_snapshot(path, inputs, outputs)
    loaded_inputs, loaded_outputs = load_snapshot(path)
"""

from __future__ import annotations

import copy
import subprocess
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

ARTIFACTS_REPO = "https://github.com/yueming-yuan/miles-artifacts.git"
ARTIFACTS_CACHE = Path.home() / ".cache" / "miles-test-artifacts"


# ---------------------------------------------------------------------------
# ParallelState (single-process, no distributed)
# ---------------------------------------------------------------------------


def make_parallel_state(is_pp_last_stage: bool = True) -> ParallelState:
    # [FUNCTION DOC] HELPER: Builds a single-process parallel state where all axes are trivial
    # (rank 0, size 1, no group) except: the tensor-parallel axis uses the world process group when
    # distributed is initialized (needed by the fused vocab-parallel cross entropy), else none; and
    # when is_pp_last_stage is false the pipeline axis has size 2 (rank 0). Records the last-stage
    # flag, installs the state as the process-global parallel state (side effect), and returns it.
    def _trivial_group() -> GroupInfo:
        # [FUNCTION DOC] HELPER: Returns a rank 0, size 1 group info with no process group.
        return GroupInfo(rank=0, size=1, group=None)

    # The fused vocab-parallel CE needs a real group object; use the 1-rank
    # world group when the test initialized one.
    tp_group = dist.group.WORLD if dist.is_initialized() else None
    state = ParallelState(
        intra_dp=_trivial_group(),
        intra_dp_cp=_trivial_group(),
        cp=_trivial_group(),
        tp=GroupInfo(rank=0, size=1, group=tp_group),
        pp=_trivial_group() if is_pp_last_stage else GroupInfo(rank=0, size=2, group=None),
        ep=_trivial_group(),
        etp=_trivial_group(),
        indep_dp=_trivial_group(),
        is_pp_last_stage=is_pp_last_stage,
    )
    set_parallel_state(state)
    return state


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

_ARGS_DEFAULTS = dict(
    # get_responses / get_log_probs_and_entropy / get_values
    qkv_format="thd",
    rollout_temperature=1.0,
    allgather_cp=False,
    log_probs_chunk_size=-1,
    true_on_policy_mode=True,
    debug_unified_grad_fused_logprob=False,
    # compute_advantages_and_returns
    advantage_estimator="grpo",
    use_rollout_logprobs=False,
    use_sampling_support_replay=False,
    skip_actor_forward_only=False,
    kl_coef=0.1,
    kl_loss_type="k1",
    gamma=1.0,
    lambd=0.95,
    normalize_advantages=False,
    # on-policy distillation (OPD); orthogonal to the advantage estimator
    use_opd=False,
    opd_type=None,
    opd_kl_coef=1.0,
    # policy_loss_function
    loss_type="policy_loss",
    eps_clip=0.2,
    eps_clip_high=0.2,
    entropy_coef=0.01,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    use_unbiased_kl=False,
    use_tis=False,
    get_mismatch_metrics=False,
    tis_clip=1.5,
    tis_clip_low=0.5,
    custom_tis_function_path=None,
    custom_pg_loss_reducer_function_path=None,
    use_opsm=False,
    opsm_delta=0.1,
    calculate_per_token_loss=False,
    # value_loss_function
    value_clip=0.2,
    # loss_function dispatcher
    global_batch_size=1,  # overridden by make_inputs
    use_dynamic_global_batch_size=False,
    recompute_loss_function=False,
)


def make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns an argument namespace made from the module defaults updated
    # with the keyword overrides (unknown keys are accepted and added).
    # KNOWN ISSUE: a comment on the defaults says the global batch size is overridden by the input
    # generator, but the generator never changes it; the default of 1 stays unless overridden here.
    d = {**_ARGS_DEFAULTS, **overrides}
    return Namespace(**d)


def args_to_dict(args: Namespace) -> dict:
    # [FUNCTION DOC] HELPER: Returns the namespace's attribute dictionary itself (not a copy, so
    # later mutation of the namespace is visible through it).
    return vars(args)


def args_from_dict(d: dict) -> Namespace:
    # [FUNCTION DOC] HELPER: Rebuilds a namespace from a saved dictionary, with module defaults
    # filling keys missing from the dictionary (so older snapshots lacking newer args still load).
    return Namespace(**{**_ARGS_DEFAULTS, **d})


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def make_inputs(
    seed: int,
    batch_size: int,
    prompt_lens: list[int],
    response_lens: list[int],
    vocab_size: int,
    args: Namespace,
) -> dict:
    # [FUNCTION DOC] HELPER: Builds a dictionary of deterministic inputs from a seeded CPU random
    # generator. Requires (assertion) that prompt and response length lists both have batch_size
    # entries. Total length per sample is prompt plus response. Per-sample tensors in draw order:
    # token ids (uniform integers below vocab size, total length), policy log-probs, reference
    # log-probs, rollout log-probs (each normal samples scaled by 2 minus 3, response length),
    # loss masks (all ones), rewards (python floats drawn from the generator), values, advantages,
    # returns (standard normal, response length), teacher log-probs (like log-probs but total
    # length). Logits: when the args qkv format is padded (bshd), max sequence length is the
    # largest total and logits shapes are batch by max length by vocab (policy) and by 1 (value),
    # max_seq_lens lists that max per sample; otherwise (packed) shapes are 1 by sum of totals by
    # vocab / 1 and max_seq_lens is none. Returns a dictionary with the seed, sizes, argument
    # dictionary, total lengths, max sequence lengths and all tensors above. Ordering of draws
    # matters for reproducing saved snapshots.
    """Generate deterministic test inputs from a seed."""
    assert len(prompt_lens) == len(response_lens) == batch_size
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]

    g = torch.Generator()
    g.manual_seed(seed)

    def randn(shape):
        # [FUNCTION DOC] HELPER: Draws a float32 standard normal tensor of the shape from the seeded
        # generator.
        return torch.randn(shape, generator=g, dtype=torch.float32)

    def randint(low, high, shape):
        # [FUNCTION DOC] HELPER: Draws an int64 tensor of the shape with values in [low, high) from
        # the seeded generator.
        return torch.randint(low, high, shape, generator=g, dtype=torch.int64)

    # Per-sample tensors
    unconcat_tokens = [randint(0, vocab_size, (tl,)) for tl in total_lens]
    log_probs = [randn((rl,)) * 2 - 3 for rl in response_lens]  # roughly [-5, -1]
    ref_log_probs = [randn((rl,)) * 2 - 3 for rl in response_lens]
    rollout_log_probs = [randn((rl,)) * 2 - 3 for rl in response_lens]
    loss_masks = [torch.ones(rl, dtype=torch.float32) for rl in response_lens]
    rewards = [torch.randn(1, generator=g).item() for _ in range(batch_size)]
    values = [randn((rl,)) for rl in response_lens]
    advantages = [randn((rl,)) for rl in response_lens]
    returns = [randn((rl,)) for rl in response_lens]
    teacher_log_probs = [randn((tl,)) * 2 - 3 for tl in total_lens]

    # Logits shape depends on qkv_format:
    #   thd:  [1, sum(total_lens), V]  — packed sequences
    #   bshd: [batch_size, max_seq_len, V] — padded per sample
    if args.qkv_format == "bshd":
        max_seq_len = max(total_lens)
        max_seq_lens = [max_seq_len] * batch_size
        policy_logits = randn((batch_size, max_seq_len, vocab_size))
        value_logits = randn((batch_size, max_seq_len, 1))
    else:
        max_seq_lens = None
        total_tokens = sum(total_lens)
        policy_logits = randn((1, total_tokens, vocab_size))
        value_logits = randn((1, total_tokens, 1))

    return dict(
        seed=seed,
        batch_size=batch_size,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        vocab_size=vocab_size,
        args_dict=args_to_dict(args),
        total_lens=total_lens,
        max_seq_lens=max_seq_lens,
        unconcat_tokens=unconcat_tokens,
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        rewards=rewards,
        values=values,
        advantages=advantages,
        returns=returns,
        teacher_log_probs=teacher_log_probs,
        policy_logits=policy_logits,
        value_logits=value_logits,
    )


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def make_rollout_data(inputs: dict) -> dict:
    # [FUNCTION DOC] HELPER: Builds the rollout data dictionary for the advantage computation from
    # generated inputs: deep copies of log-probs, reference log-probs, rollout log-probs, values,
    # loss masks and teacher log-probs, plain list copies of rewards, response lengths and total
    # lengths; adds max sequence lengths (copied) only when the inputs have them.
    """Build the rollout_data dict consumed by compute_advantages_and_returns."""
    d = dict(
        log_probs=deep_clone(inputs["log_probs"]),
        ref_log_probs=deep_clone(inputs["ref_log_probs"]),
        rollout_log_probs=deep_clone(inputs["rollout_log_probs"]),
        rewards=list(inputs["rewards"]),
        values=deep_clone(inputs["values"]),
        response_lengths=list(inputs["response_lens"]),
        loss_masks=deep_clone(inputs["loss_masks"]),
        total_lengths=list(inputs["total_lens"]),
        teacher_log_probs=deep_clone(inputs["teacher_log_probs"]),
    )
    if inputs.get("max_seq_lens") is not None:
        d["max_seq_lens"] = list(inputs["max_seq_lens"])
    return d


def make_batch(inputs: dict, loss_type: str) -> dict:
    # [FUNCTION DOC] HELPER: Builds the batch dictionary for loss functions from generated inputs:
    # deep copies of tokens, loss masks, log-probs, reference and rollout log-probs, advantages and
    # returns; list copies of response and total lengths; max sequence lengths when present; and
    # the values only when the loss type is the value loss. Rewards, teacher log-probs and logits
    # are not included.
    """Build the RolloutBatch dict consumed by loss functions."""
    batch = dict(
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        response_lengths=list(inputs["response_lens"]),
        total_lengths=list(inputs["total_lens"]),
        loss_masks=deep_clone(inputs["loss_masks"]),
        log_probs=deep_clone(inputs["log_probs"]),
        ref_log_probs=deep_clone(inputs["ref_log_probs"]),
        rollout_log_probs=deep_clone(inputs["rollout_log_probs"]),
        advantages=deep_clone(inputs["advantages"]),
        returns=deep_clone(inputs["returns"]),
    )
    if inputs.get("max_seq_lens") is not None:
        batch["max_seq_lens"] = list(inputs["max_seq_lens"])
    if loss_type == "value_loss":
        batch["values"] = deep_clone(inputs["values"])
    return batch


# ---------------------------------------------------------------------------
# Deep clone / comparison
# ---------------------------------------------------------------------------


def deep_clone(obj):
    # [FUNCTION DOC] HELPER: Recursively copies a structure: tensors are cloned, dictionaries are
    # rebuilt with cloned values, lists and tuples are rebuilt (keeping their type) from cloned
    # elements, anything else gets a shallow copy. Returns the new structure.
    """Recursively clone tensors so in-place ops don't corrupt shared state."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: deep_clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [deep_clone(x) for x in obj]
        return type(obj)(cloned)
    return copy.copy(obj)


# torch.exp(float32) vectorized path can differ by 1 ULP across torch builds
# with the same version string (different CPU SIMD / MKL link). Measured drift
# between the snapshot env and radixark/miles:dev on ion-user-7 H200: max 1.91e-06.
# 2e-06 absorbs it. All other ops (cat, sub, mul, clamp, softmax, etc.) are bit-exact.
TENSOR_ATOL = 2e-6


def assert_outputs_equal(actual, expected, path: str = "root"):
    # [FUNCTION DOC] HELPER: Recursively asserts that actual equals expected, using a path string in
    # failure messages. Tensors: actual must be a tensor with identical shape and dtype, and the
    # maximum absolute elementwise difference (zero when empty) must not exceed the tolerance
    # (2e-6), else an assertion error reports max and mean difference. Dictionaries: must be
    # dictionaries with identical key sets (extra and missing keys reported), compared per key.
    # Lists and tuples: either kind accepted for either, same length, compared per index. Anything
    # else: compared for equality. Raises assertion errors; returns nothing.
    # KNOWN ISSUE: the docstring says "<= TENSOR_ATOL" but the code only fails when the maximum
    # difference exceeds it, which is equivalent; non-finite values (NaN) would not fail the
    # comparison because a NaN maximum compares false against the tolerance.
    """Recursively compare outputs. Tensors compared with `<= TENSOR_ATOL`."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected Tensor, got {type(actual)}"
        assert actual.shape == expected.shape, f"{path}: shape mismatch {actual.shape} vs {expected.shape}"
        assert actual.dtype == expected.dtype, f"{path}: dtype mismatch {actual.dtype} vs {expected.dtype}"
        diff = (actual - expected).abs()
        max_diff = diff.max().item() if diff.numel() else 0.0
        if max_diff > TENSOR_ATOL:
            raise AssertionError(
                f"{path}: tensors differ beyond TENSOR_ATOL={TENSOR_ATOL:.0e}. "
                f"max_diff={max_diff:.2e}, mean_diff={diff.mean().item():.2e}"
            )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"{path}: key mismatch. extra={set(actual.keys()) - set(expected.keys())}, "
            f"missing={set(expected.keys()) - set(actual.keys())}"
        )
        for k in expected:
            assert_outputs_equal(actual[k], expected[k], path=f"{path}[{k!r}]")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"{path}: expected list/tuple, got {type(actual)}"
        assert len(actual) == len(expected), f"{path}: length mismatch {len(actual)} vs {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_outputs_equal(a, e, path=f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


# ---------------------------------------------------------------------------
# Snapshot save / load
# ---------------------------------------------------------------------------


def ensure_snapshot_dir(snapshot_dir: Path) -> Path:
    # [FUNCTION DOC] HELPER: Locates snapshot files. Returns the given directory if it exists and
    # holds any snapshot (.pt) file; else returns the loss_snapshots folder of the cached clone if
    # it holds any; else creates the cache directory, shallow-clones (depth 1) the artifacts repo
    # into it with a subprocess (network access; raises a called-process error on failure), asserts
    # the loss_snapshots folder exists, and returns it.
    """Return a directory containing .pt snapshots.

    If *snapshot_dir* already has .pt files, return it as-is.
    Otherwise, shallow-clone yueming-yuan/miles-artifacts into a cache
    directory and return the path to loss_snapshots/ inside the clone.
    """
    if snapshot_dir.exists() and any(snapshot_dir.glob("*.pt")):
        return snapshot_dir
    repo_dir = ARTIFACTS_CACHE / "repo"
    result_dir = repo_dir / "loss_snapshots"
    if result_dir.exists() and any(result_dir.glob("*.pt")):
        return result_dir
    ARTIFACTS_CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", ARTIFACTS_REPO, str(repo_dir)],
        check=True,
    )
    assert result_dir.exists(), f"Expected {result_dir} after clone"
    return result_dir


def save_snapshot(path: str | Path, inputs: dict, outputs: dict) -> None:
    # [FUNCTION DOC] HELPER: Creates the parent directory if needed and serializes a dictionary with
    # the inputs and outputs to the path with torch's saver. Returns nothing.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"inputs": inputs, "outputs": outputs}, path)


def load_snapshot(path: str | Path) -> tuple[dict, dict]:
    # [FUNCTION DOC] HELPER: Loads a snapshot file with torch's loader (full unpickling, weights-only
    # disabled, so only trusted files should be loaded) and returns the (inputs, outputs) pair.
    data = torch.load(path, weights_only=False)
    return data["inputs"], data["outputs"]
