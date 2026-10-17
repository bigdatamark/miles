# [MODULE DOC]
# Log probabilities (and optionally entropies) of chosen token ids computed from logits whose
# vocabulary dimension is sharded across a tensor-parallel group, communicating only per-row scalars.
# Two custom autograd functions: one normalizing over the real (unpadded) full vocabulary, one
# normalizing over only the supplied candidate support (sampling-support replay). The public entry
# points validate inputs and process rows in chunks.
"""Selected-token log-probabilities from vocabulary-sharded logits."""

from typing import Any

import torch
import torch.distributed as dist


class _SelectedLogProbs(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function giving, for logits [rows, local vocab] and token ids
    # [rows, K] (global ids, -1 means padding), the log softmax at those ids over the real
    # vocabulary, plus per-row entropy. Backward writes the gradient directly into the local slice.
    """Vocabulary-sharded log-softmax gather with a replicated loss on TP ranks.

    Only selected logits and normalization scalars are communicated. Backward
    computes each rank's vocabulary slice directly; reducing identical output
    gradients across TP ranks would incorrectly multiply the gradient by TP.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        group: dist.ProcessGroup | None,
        vocab_size: int,
        with_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] Inputs: local logits, global token ids, group (none means unsharded), the
        # real vocab size and an entropy flag. Steps: this rank owns ids in [rank*width,
        # (rank+1)*width); ids are valid when non-negative and local when owned; columns beyond the
        # real vocab (padding) are set to negative infinity; row max (max all-reduce); exp of shifted
        # logits and their row sum (sum all-reduce); the selected logit is gathered for locally owned
        # ids (zero otherwise) and summed across ranks; probabilities = exp / sum; entropy (if
        # requested) = -sum p log p with zero for p = 0, summed across ranks. Returns (log probs
        # [rows, K] = selected - max - log sum, zero for padding ids; entropy [rows], zeros when not
        # requested). Saves probabilities and index masks for backward. Collectives on the group.
        rank = dist.get_rank(group) if group is not None else 0
        width = logits.shape[-1]
        valid = token_ids >= 0
        local_ids = token_ids - rank * width
        local = valid & (local_ids >= 0) & (local_ids < width)
        local_ids = local_ids.clamp(0, width - 1)
        padded = torch.arange(width, device=logits.device) + rank * width >= vocab_size
        work = logits.masked_fill(padded, -torch.inf)
        maximum = work.amax(-1, keepdim=True)
        if group is not None:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        probabilities = (work - maximum).exp()
        denominator = probabilities.sum(-1, keepdim=True)
        selected = torch.where(local, work.gather(-1, local_ids), 0.0)
        if group is not None:
            dist.all_reduce(denominator, group=group)
            dist.all_reduce(selected, group=group)
        probabilities.div_(denominator)
        entropy = probabilities.new_zeros(probabilities.size(0))
        if with_entropy:
            entropy = -(probabilities * torch.where(probabilities > 0, probabilities.log(), 0.0)).sum(-1)
            if group is not None:
                dist.all_reduce(entropy, group=group)
        ctx.with_entropy = with_entropy
        ctx.save_for_backward(probabilities, local_ids, local, valid, entropy)
        return torch.where(valid, selected - maximum - denominator.log(), 0.0), entropy

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, grad_entropy: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        # [FUNCTION DOC] Gradient w.r.t. local logits: -probabilities * (sum over K of upstream
        # grad, zeroed at padding ids) plus the upstream grad scattered onto locally owned selected
        # columns; if entropy was computed, subtracts entropy-grad * p * (log p + entropy). No
        # communication, so each rank's slice gets the full (not TP-multiplied) gradient. Returns the
        # logits gradient and none for the other inputs.
        probabilities, local_ids, local, valid, entropy = ctx.saved_tensors
        grad = grad_output.masked_fill(~valid, 0.0)
        grad_logits = probabilities * (-grad.sum(-1, keepdim=True))
        grad_logits.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0.0))
        if ctx.with_entropy:
            logp = torch.where(probabilities > 0, probabilities.log(), 0.0)
            grad_logits -= grad_entropy.unsqueeze(-1) * probabilities * (logp + entropy.unsqueeze(-1))
        return grad_logits, None, None, None, None


class _SupportLogProbs(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for sampling-support replay. Column 0 of the token ids is
    # the sampled token and the remaining columns are the complete support; the distribution is
    # renormalized over the support only (not the full vocab). Saves only K-sized probabilities.
    """Score the sampled token and complete support without saving vocabulary-sized probabilities."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        group: dist.ProcessGroup | None,
        temperature: float,
        with_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] Inputs: local logits, global ids [rows, 1 + support], group, temperature and
        # an entropy flag. Gathers the logit of each locally owned id (float32, or float64 for
        # float64 logits), divides by temperature, sums across ranks. Support columns are valid when
        # id is non-negative; a row has support if any support column is valid. The normalizer is the
        # log-sum-exp over valid support columns (0 when none). Log probs = selected - normalizer,
        # zero where the id is invalid or the row has no support. Entropy (if requested) is the
        # entropy of the support distribution, else zeros. Returns (log probs [rows, 1 + support],
        # entropy [rows]). Saves support probs for backward. One sum collective on the group.
        rank = dist.get_rank(group) if group is not None else 0
        width = logits.size(-1)
        valid = token_ids >= 0
        local_ids = token_ids - rank * width
        local = valid & (local_ids >= 0) & (local_ids < width)
        local_ids = local_ids.clamp(0, width - 1)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        selected = torch.where(local, logits.gather(-1, local_ids).to(dtype), 0.0) / temperature
        if group is not None:
            dist.all_reduce(selected, group=group)
        head_valid = valid[:, 1:]
        has_support = head_valid.any(-1, keepdim=True)
        head = selected[:, 1:].masked_fill(~head_valid, -torch.inf)
        head = torch.where(has_support, head, 0.0)
        logp = selected - torch.logsumexp(head, dim=-1, keepdim=True)
        logp = torch.where(valid & has_support, logp, 0.0)
        probabilities = torch.where(head_valid, logp[:, 1:].exp(), 0.0)
        entropy = logits.new_zeros(logits.size(0), dtype=dtype)
        if with_entropy:
            entropy = -(probabilities * torch.where(probabilities > 0, logp[:, 1:], 0.0)).sum(-1)
        ctx.logits_shape = logits.shape
        ctx.temperature = temperature
        ctx.with_entropy = with_entropy
        ctx.save_for_backward(probabilities, local_ids, local, valid & has_support, entropy)
        return logp, entropy

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        # [FUNCTION DOC] Builds a zero gradient of the logits' shape, scatters into it the upstream
        # gradient at locally owned columns (sampled and support), plus the softmax term for the
        # support columns (-probs times the summed upstream gradient, with the entropy term when
        # enabled), then divides by the temperature. Returns the logits gradient and none for the
        # other inputs. No communication.
        probabilities, local_ids, local, valid, entropy = ctx.saved_tensors
        grad = grad_output.masked_fill(~valid, 0.0)
        head_grad = probabilities * (-grad.sum(-1, keepdim=True))
        if ctx.with_entropy:
            logp = torch.where(probabilities > 0, probabilities.log(), 0.0)
            head_grad -= grad_entropy.unsqueeze(-1) * probabilities * (logp + entropy.unsqueeze(-1))
        grad_logits = grad.new_zeros(ctx.logits_shape)
        grad_logits.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0.0))
        grad_logits.scatter_add_(-1, local_ids[:, 1:], head_grad.masked_fill(~local[:, 1:], 0.0))
        grad_logits.div_(ctx.temperature)
        return grad_logits, None, None, None, None


def selected_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
) -> torch.Tensor:
    # [FUNCTION DOC] Convenience wrapper returning only the log probs of the combined function
    # (no entropy), with the same arguments minus the entropy and support options.
    """Log-probabilities at global IDs [T, K] from local logits [T, V/TP].

    -1 is a padding ID and produces zero with zero gradient. The true vocabulary
    size excludes Megatron's padded vocabulary entries from normalization.
    """
    return selected_log_probs_and_entropy(
        logits, token_ids, group=group, vocab_size=vocab_size, temperature=temperature, chunk_size=chunk_size
    )[0]


def selected_log_probs_and_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
    with_entropy: bool = False,
    sampling_support: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Public entry. Defaults: vocab size = local width times group size (no
    # padding); the group size comes from the process group (1 when none). Raises a value error
    # for non-positive temperature, vocab size outside (0, padded vocab], inputs not 2-D or with
    # different row counts, or token ids below -1 or at/above the vocab size. With zero rows returns
    # empty-shaped results derived from the logits (a sum of the logits expanded to the id shape, and
    # a per-row sum). Otherwise splits rows into chunks of chunk_size (non-positive means a single
    # chunk), and for each chunk uses the support function (when sampling_support) or the full-vocab
    # function (logits cast to float32/float64 and divided by temperature), then concatenates.
    # Returns (log probs [rows, K], entropy [rows]); entropy is zeros unless requested.
    """Selected logprobs and optional entropy of the same unpadded distribution.

    With sampling_support, column zero is the sampled token and remaining
    columns are the complete support. Rows without candidates contribute zero.
    """
    size = dist.get_world_size(group) if group is not None else 1
    vocab_size = vocab_size if vocab_size is not None else logits.size(-1) * size
    if temperature <= 0 or not 0 < vocab_size <= logits.size(-1) * size:
        raise ValueError("Score centering needs a positive temperature and valid vocabulary size")
    if token_ids.ndim != 2 or logits.ndim != 2 or logits.size(0) != token_ids.size(0):
        raise ValueError("Expected logits [T, V/TP] and selected token IDs [T, K]")
    if ((token_ids < -1) | (token_ids >= vocab_size)).any():
        raise ValueError("Score-centering token ID is outside the model vocabulary")
    if logits.size(0) == 0:
        return logits.sum(-1, keepdim=True).expand_as(token_ids), logits.sum(-1)
    chunk_size = chunk_size if chunk_size > 0 else logits.size(0)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    chunks = []
    for chunk, ids in zip(logits.split(chunk_size), token_ids.split(chunk_size), strict=True):
        if sampling_support:
            chunks.append(_SupportLogProbs.apply(chunk, ids, group, temperature, with_entropy))
        else:
            chunks.append(_SelectedLogProbs.apply(chunk.to(dtype) / temperature, ids, group, vocab_size, with_entropy))
    return torch.cat([chunk[0] for chunk in chunks]), torch.cat([chunk[1] for chunk in chunks])
