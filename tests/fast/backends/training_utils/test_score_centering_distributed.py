# [MODULE DOC] Four-process tests of tensor parallel 2 by context parallel 2 parity for the score
# centering loss: the selected log-prob/entropy helper with a vocabulary sharded across the tensor
# parallel pair, and the full loss function across layouts (packed, padded, and the allgather
# variants of each), importance modes (none, tis, mis) and sampling-support replay on/off with a KL
# loss on/off, each compared against a dense single-device reference (loss and local gradient).
# Runs on CPU with Gloo by default (spawned in a normal test), and with NCCL on four GPUs only when
# an opt-in environment variable is set and at least four devices exist.
"""TP=2, CP=2 parity on CPU/Gloo and opt-in four-GPU NCCL."""

import os
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from miles.backends.training_utils.data.context_parallel import (
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
    slice_with_cp,
)
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.score_centering import (
    ScoreCenteringInputs,
    score_centering_loss,
    selected_log_probs_and_entropy,
)
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.sampling_mask import RolloutSamplingMask


def _layout(parts: list[torch.Tensor], args: Namespace, cp_rank: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts full per-sample tensors (rows by width) into the local tensor
    # one context-parallel rank holds in the configured layout. Allgather layout, padded format:
    # each sample is zero-padded along rows to 12, split in two contiguous halves, the rank's half
    # taken, samples stacked. Allgather, packed format: samples concatenated, padded with one zero
    # row when the total row count is odd (none when even), split into two halves, the rank's half
    # given a leading batch axis. Otherwise (zigzag): each
    # sample sliced for this rank with the context-parallel slicer using padded max length 12, then
    # stacked (padded format) or concatenated with a leading batch axis (packed). Returns it.
    if args.allgather_cp:
        if args.qkv_format == "bshd":
            return torch.stack([F.pad(part, (0, 0, 0, 12 - part.size(0))).chunk(2)[cp_rank] for part in parts])
        full = torch.cat(parts)
        full = F.pad(full, (0, 0, 0, full.size(0) % 2))
        return full.chunk(2)[cp_rank].unsqueeze(0)
    pieces = [slice_with_cp(part, 0, args.qkv_format, 12) for part in parts]
    return torch.stack(pieces) if args.qkv_format == "bshd" else torch.cat(pieces).unsqueeze(0)


def _check_selected(tp: GroupInfo, dtype: torch.dtype, device: torch.device) -> None:
    # [FUNCTION DOC] HELPER: Checks the selected log-prob and entropy helper under tensor
    # parallelism. A seeded full logits tensor (4 rows by 8, given dtype and device, requiring
    # grad) is split by columns into 2 shards of 4; this rank's shard is the local input. Ids are 4
    # by 4 with -1 padding; vocabulary size 7, temperature 0.7, chunk size 2, entropy on. Compares
    # entropy and selected log-probs (tolerance 1e-6) with a dense reference over the full logits
    # (sliced to the vocabulary); then backpropagates identical weighted objectives (random weights
    # minus 0.2 times entropy) through both and compares the local gradient with the matching column
    # slice of the reference gradient (tolerance 0.01 for bf16, else 1e-6).
    full = (
        torch.randn(4, 8, generator=torch.Generator().manual_seed(31)).to(device=device, dtype=dtype).requires_grad_()
    )
    local = full.detach()[:, tp.rank * 4 : (tp.rank + 1) * 4].clone().requires_grad_()
    ids = torch.tensor([[0, 3, 6, -1], [1, 4, 4, 1], [6, 3, -1, -1], [5, 2, 4, 6]], device=device)
    actual, entropy = selected_log_probs_and_entropy(
        local, ids, group=tp.group, vocab_size=7, temperature=0.7, chunk_size=2, with_entropy=True
    )
    logp = (full.float()[:, :7] / 0.7).log_softmax(-1)
    reference = logp.gather(-1, ids.clamp_min(0)).masked_fill(ids < 0, 0)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(entropy, expected_entropy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    weights = torch.randn(4, 4, generator=torch.Generator().manual_seed(41)).to(device)
    ((actual * weights).sum() - 0.2 * entropy.sum()).backward()
    ((reference * weights).sum() - 0.2 * expected_entropy.sum()).backward()
    expected = full.grad[:, tp.rank * 4 : (tp.rank + 1) * 4]
    torch.testing.assert_close(
        local.grad,
        expected,
        atol=0.01 if dtype == torch.bfloat16 else 1e-6,
        rtol=0.01 if dtype == torch.bfloat16 else 1e-6,
    )


def _check_loss(
    tp: GroupInfo,
    cp: GroupInfo,
    layout: str,
    mode: str,
    device: torch.device,
    replay: bool = False,
    use_kl: bool = True,
) -> None:
    # [FUNCTION DOC] HELPER: Checks the full score centering loss function for one configuration.
    # Inputs: tensor and context parallel group infos, a layout name (thd, bshd, allgather,
    # bshd_allgather), the importance mode, device, a replay flag (sampling-support replay: the
    # head becomes the sampling support and train log-probs are renormalized over it) and a KL
    # flag. Builds args (score centering, temperature 0.7, chunk 2, vocab 7, tis clip 2, mis band
    # 0.5 to 5, entropy coefficient 0.03, KL k2 with coefficient 0.1). Three samples with total
    # lengths 5, 9, 3 and responses 2, 3, 0 (one has an empty response), random logits (width 8, 7
    # real vocabulary entries), tokens, partial masks, advantages, and rollout distributions whose
    # top-3 form the head (in replay mode the sampled token is the head's top entry, heads are
    # renormalized, and one head slot is invalidated with -1 / 0). Reference path: dense per-sample
    # log-softmax at the response positions, optional support renormalization, the dense loss via the
    # score centering helper minus entropy coefficient times entropy (head-restricted in replay),
    # masked mean per sample, summed; backward gives reference gradients. Under test: this rank's
    # local logits (layout slice, this tensor shard of columns), a batch with lengths, tokens, masks,
    # rollout top-k ids and log-probs (replay uses lengths and sampling-mask ids/offsets instead),
    # context-parallel-sliced rollout log-probs and advantages, and reference log-probs equal to the
    # actor's own (so the KL loss must be exactly zero when enabled). Calls the loss function; asserts
    # KL metric zero, local gradient equals the layout/shard slice of the reference gradient (atol
    # 2e-6, rtol 2e-5), the loss summed over the context-parallel group equals the reference loss, and
    # all metrics are finite.
    args = Namespace(
        loss_type="score_centering",
        qkv_format="bshd" if layout in ("bshd", "bshd_allgather") else "thd",
        allgather_cp=layout in ("allgather", "bshd_allgather"),
        true_on_policy_mode=False,
        rollout_temperature=0.7,
        log_probs_chunk_size=2,
        vocab_size=7,
        score_centering_is=mode,
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        entropy_coef=0.03,
        observe_training_entropy=True,
        use_kl_loss=use_kl,
        use_sampling_support_replay=replay,
        use_unbiased_kl=False,
        kl_loss_type="k2",
        kl_loss_coef=0.1,
    )
    generator = torch.Generator().manual_seed(19)
    totals, responses = [5, 9, 3], [2, 3, 0]
    parts = [torch.randn(total, 8, generator=generator).to(device).requires_grad_() for total in totals]
    tokens = [torch.randint(0, 7, (total,), generator=generator).to(device) for total in totals]
    masks = [torch.tensor(mask, device=device) for mask in ([1, 0], [1, 1, 1], [])]
    advantages = [torch.randn(response, generator=generator).to(device) for response in responses]
    distributions = [torch.randn(response, 7, generator=generator).to(device).softmax(-1) for response in responses]
    candidates = [q.topk(3, dim=-1) for q in distributions]
    if replay:
        for token, response, head in zip(tokens, responses, candidates, strict=True):
            if response:
                token[-response:] = head.indices[:, 0]
                head.values.div_(head.values.sum(-1, keepdim=True))
        candidates[0].indices[1].fill_(-1)
        candidates[0].values[1].zero_()
    sampled_q = [
        q.gather(-1, token[-response:, None] if response else token[:0, None]).squeeze(-1).log()
        for q, token, response in zip(distributions, tokens, responses, strict=True)
    ]
    if replay:
        sampled_q = [head.values[:, 0].clamp_min(1e-30).log() for head in candidates]
    expected_loss = torch.zeros((), device=device)
    for part, token, total, response, advantage, mask, head, q_sample in zip(
        parts, tokens, totals, responses, advantages, masks, candidates, sampled_q, strict=True
    ):
        logp = (part[total - response - 1 : total - 1, :7] / 0.7).log_softmax(-1)
        sampled = token[-response:] if response else token[:0]
        valid = head.indices >= 0
        if replay:
            selected = logp.gather(-1, head.indices.clamp_min(0)).masked_fill(~valid, -torch.inf)
            normalizer = torch.logsumexp(selected, -1, keepdim=True)
            logp = logp - torch.where(valid.any(-1, keepdim=True), normalizer, 0.0)
        per_token, _ = score_centering_loss(
            ScoreCenteringInputs(
                train_log_probs=logp.gather(-1, sampled[:, None]).squeeze(-1),
                train_head_log_probs=logp.gather(-1, head.indices.clamp_min(0)),
                rollout_log_probs=q_sample,
                rollout_head_log_probs=head.values.log(),
                head_mask=valid,
                advantages=advantage,
                mode=mode,
            ),
        )
        if replay:
            head_logp = logp.gather(-1, head.indices.clamp_min(0)).masked_fill(~valid, -torch.inf)
            entropy = -(head_logp.exp() * head_logp.masked_fill(~valid, 0.0)).sum(-1)
        else:
            entropy = -(logp.exp() * logp).sum(-1)
        expected_loss = expected_loss + (
            (per_token - args.entropy_coef * entropy) * mask
        ).sum() / mask.sum().clamp_min(1)
    expected_loss.backward()
    local = (
        _layout([part.detach() for part in parts], args, cp.rank)[..., tp.rank * 4 : (tp.rank + 1) * 4]
        .clone()
        .requires_grad_()
    )
    batch = dict(
        total_lengths=totals,
        response_lengths=responses,
        unconcat_tokens=tokens,
        loss_masks=masks,
        max_seq_lens=[12] * 3 if args.qkv_format == "bshd" else None,
        rollout_topk_token_ids=[head.indices.cpu().numpy().astype("int32") for head in candidates],
        rollout_topk_log_probs=[head.values.log().cpu().numpy() for head in candidates],
        rollout_log_probs=[
            slice_log_prob_with_cp(q, t, r, args.qkv_format, 12)
            for q, t, r in zip(sampled_q, totals, responses, strict=True)
        ],
        advantages=[
            slice_log_prob_with_cp(a, t, r, args.qkv_format, 12)
            for a, t, r in zip(advantages, totals, responses, strict=True)
        ],
    )
    if replay:
        dense_ids = [torch.from_numpy(ids) for ids in batch.pop("rollout_topk_token_ids")]
        batch["rollout_topk_lengths"] = [(ids >= 0).sum(-1).to(torch.int32) for ids in dense_ids]
        supports = [
            RolloutSamplingMask.from_mask_list(
                [
                    row[row >= 0].tolist() if (row >= 0).any() else [int(token[-response + i])]
                    for i, row in enumerate(ids)
                ]
            )
            for ids, token, response in zip(dense_ids, tokens, responses, strict=True)
        ]
        batch["rollout_sampling_mask_ids"] = [mask._as_tensors()[0] for mask in supports]
        batch["rollout_sampling_mask_offsets"] = [mask._as_tensors()[1] for mask in supports]
    # A reference identical to the actor must give zero KL, including with a padded vocabulary.
    with torch.no_grad():
        batch["ref_log_probs"] = get_log_probs_and_entropy(
            local.detach(),
            args=args,
            unconcat_tokens=tokens,
            total_lengths=totals,
            response_lengths=responses,
            max_seq_lens=batch["max_seq_lens"],
        )["log_probs"]
    reduce = get_sum_of_sample_mean(
        totals, responses, masks, qkv_format=args.qkv_format, max_seq_lens=batch["max_seq_lens"]
    )
    loss, metrics = score_centering_loss_function(args, batch, local, reduce)
    if use_kl:
        torch.testing.assert_close(metrics["kl_loss"], torch.zeros_like(metrics["kl_loss"]), atol=0, rtol=0)
    loss.backward()
    expected_grad = _layout([part.grad for part in parts], args, cp.rank)[..., tp.rank * 4 : (tp.rank + 1) * 4]
    torch.testing.assert_close(local.grad, expected_grad, atol=2e-6, rtol=2e-5)
    actual_loss = loss.detach().clone()
    dist.all_reduce(actual_loss, group=cp.group)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-6, rtol=2e-5)
    assert all(torch.isfinite(value) for value in metrics.values())


def _worker(rank: int, rendezvous: str, backend: str) -> None:
    # [FUNCTION DOC] HELPER: Per-rank body spawned for 4 ranks. Single-threaded; picks the CUDA
    # device of the rank for NCCL else CPU; initializes the process group (120 second timeout) from
    # the rendezvous URL. Builds tensor parallel groups {0,1},{2,3} and context parallel groups
    # {0,2},{1,3} (tensor rank = rank mod 2, context rank = rank div 2), installs a global parallel
    # state with those and trivial other axes. Runs the selected-log-prob check for fp32 and bf16,
    # then for each of four layouts and three modes runs the loss check plain, and with replay on
    # both with and without KL. Always destroys the process group.
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        tp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        cp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        tp = GroupInfo(rank=rank % 2, size=2, group=tp_groups[rank // 2])
        cp = GroupInfo(rank=rank // 2, size=2, group=cp_groups[rank % 2])
        singleton = GroupInfo(rank=0, size=1, group=None)
        state = {name: singleton for name in ("intra_dp", "intra_dp_cp", "pp", "ep", "etp", "indep_dp")}
        set_parallel_state(ParallelState(**state, tp=tp, cp=cp))
        for dtype in (torch.float32, torch.bfloat16):
            _check_selected(tp, dtype, device)
        for layout in ("thd", "bshd", "allgather", "bshd_allgather"):
            for mode in ("none", "tis", "mis"):
                _check_loss(tp, cp, layout, mode, device)
                for use_kl in (False, True):
                    _check_loss(tp, cp, layout, mode, device, replay=True, use_kl=use_kl)
    finally:
        dist.destroy_process_group()


def test_tensor_and_context_parallel_gradients(tmp_path: Path) -> None:
    # [FUNCTION DOC] Test: spawns 4 processes running the worker with Gloo on CPU and a file-URI
    # rendezvous in the temp directory; failures in any process propagate.
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), "gloo"), nprocs=4, join=True)


@pytest.mark.skipif(
    os.environ.get("MILES_TEST_CUDA_DISTRIBUTED") != "1" or torch.cuda.device_count() < 4,
    reason="opt-in four-GPU NCCL parity",
)
def test_tensor_and_context_parallel_cuda_gradients(tmp_path: Path) -> None:
    # [FUNCTION DOC] Test (skipped unless the opt-in environment variable equals 1 and at least 4
    # GPUs exist): same as the CPU test but with the NCCL backend.
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), "nccl"), nprocs=4, join=True)
