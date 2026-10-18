# [MODULE DOC]
# Test that the loss mask produced by the miles data pipeline lines up with the targets used by
# the real multi-token-prediction (MTP) loss function of Megatron. Skipped when the optional
# Megatron MTP and packed-sequence modules are not installed. Runs on CPU by faking the parallel
# state (single rank, no group) and neutralizing the CUDA device and tensor transfer calls.
"""Exercise the Miles mask through the real, optional Megatron MTP loss consumer."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState

mtp = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
packed_seq = pytest.importorskip("megatron.core.packed_seq_params")


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("all_masked", [False, True])
@pytest.mark.parametrize("per_token_loss", [False, True])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_actual_mtp_loss_selects_intended_targets(
    monkeypatch: pytest.MonkeyPatch,
    num_layers: int,
    all_masked: bool,
    per_token_loss: bool,
    qkv_format: str,
) -> None:
    # [FUNCTION DOC] Parametrized over 1 or 2 MTP layers, loss masks zeroed or not, per-token
    # versus per-sequence loss normalization, and packed ("thd") versus padded ("bshd") layout.
    # Setup: single-rank fake parallel state patched into the rollout-data and context-parallel
    # modules; CUDA current device reported as CPU and the tensor cuda method made the identity;
    # the MTP loss auto-scaler's backward scale set to a tensor 1.0 (all undone by monkeypatch).
    # Data: two samples of 12 and 10 tokens with response masks of lengths 9 and 7 (multiplied by
    # zero when all are masked), maximum sequence length 24 each. A batch is built through the
    # miles batch builder with micro batch size 2 and pad multiplier 8. The model's output layer
    # is replaced by a fake that records the labels it receives and returns a ones-valued loss
    # requiring gradients per label; the actual Megatron rolling, masking, normalization and
    # gradient injection run. Hidden states are zeros of (layers + 1) times sequence length rows.
    # The expected targets are all response-token ids selected by the mask, in sample order. The
    # MTP loss processing is called with no labels (so Megatron derives them from the input ids),
    # the batch's input loss mask, and, for the packed layout, packed sequence params built from
    # the batch's cumulative lengths and max length. Then the summed output is back-propagated.
    # Asserts, for every recorded depth: the gradient exists and is finite, and the labels at
    # positions with nonzero gradient equal exactly the expected targets; and that the loss mask
    # passed in was not modified in place.
    group = GroupInfo(rank=0, size=1, group=None)
    state = ParallelState(
        intra_dp=group, intra_dp_cp=group, cp=group, tp=group, pp=group, ep=group, etp=group, indep_dp=group
    )
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))

    token_ids = [torch.arange(1, 13), torch.arange(101, 111)]
    mask_values = [[0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 1]]
    masks = [torch.tensor(values, dtype=torch.int) * int(not all_masked) for values in mask_values]
    rollout = {
        "tokens": token_ids,
        "loss_masks": masks,
        "total_lengths": [len(t) for t in token_ids],
        "response_lengths": [len(mask) for mask in masks],
        "max_seq_lens": [24, 24],
    }
    batch = data_utils.get_batch(
        data_utils.DataIterator(rollout, micro_batch_size=2),
        list(rollout),
        pad_multiplier=8,
        qkv_format=qkv_format,
    )
    targets = []
    losses = []

    def output_layer(hidden: torch.Tensor, *, labels: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Fake output layer. Ignores the hidden states and extra keyword
        # arguments; appends the labels to the enclosing target list, creates a float32 ones
        # tensor shaped like the labels that requires gradients, appends it to the enclosing loss
        # list and returns it.
        # Only the expensive model projection/CE is replaced. Real Megatron
        # rolling, masking, normalization and autograd injection run unchanged.
        targets.append(labels)
        loss = torch.ones_like(labels, dtype=torch.float32, requires_grad=True)
        losses.append(loss)
        return loss

    # Every supervised response token must be an MTP target, at every depth.
    expected_targets = []
    for tokens, mask in zip(token_ids, masks, strict=True):
        ids = tokens.tolist()
        response_ids = ids[len(ids) - len(mask) :]
        expected_targets.extend(
            token_id for token_id, selected in zip(response_ids, mask.tolist(), strict=True) if selected
        )
    config = SimpleNamespace(
        mtp_num_layers=num_layers,
        mtp_detach_heads=False,
        calculate_per_token_loss=per_token_loss,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="linear",
        mtp_loss_scaling_factor=0.2,
    )
    mask = batch["input_loss_masks"]
    original_mask = mask.clone()
    batch_size, seq_len = batch["tokens"].shape
    hidden = torch.zeros((num_layers + 1) * seq_len, batch_size, 1, requires_grad=True)
    packed_params = None
    if qkv_format == "thd":
        packed_params = packed_seq.PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=batch["cu_seqlens"],
            cu_seqlens_kv=batch["cu_seqlens"],
            max_seqlen_q=batch["max_seqlen"],
            max_seqlen_kv=batch["max_seqlen"],
        )
    # Miles always calls Megatron with labels=None, so Megatron derives the
    # labels from input_ids and rolls the input-aligned mask along with them.
    output = mtp.process_mtp_loss(
        hidden_states=hidden,
        labels=None,
        loss_mask=mask,
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=False,
        is_training=False,
        compute_language_model_loss=None,
        config=config,
        cp_group=None,
        tp_group=None,
        packed_seq_params=packed_params,
        input_ids=batch["tokens"],
    )
    output.sum().backward()
    for target, loss in zip(targets, losses, strict=True):
        assert loss.grad is not None
        assert torch.isfinite(loss.grad).all()
        assert target[loss.grad != 0].tolist() == expected_targets
    torch.testing.assert_close(mask, original_mask)
