# [MODULE DOC] Unit tests for the batch construction code of the run_megatron worker: building
# next-token labels (with and without context parallel position ids), the language-model loss
# function, and preparing a full input batch including zigzag context parallel slicing with and
# without padding. All tests run on CPU with small hand-written token sequences; context parallel
# ranks are simulated by calling the batch preparation once per rank, so no distributed setup is
# needed.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data.context_parallel import slice_with_cp
from miles.utils.debug_utils.run_megatron.worker.batch import _build_labels, loss_func, prepare_batch


def _zigzag_slice(tokens: torch.Tensor, *, cp_rank: int, cp_size: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Reference slicing of a one-dimensional tensor for one context
    # parallel rank. Calls the production zigzag slicing routine with padding value 0, the batch-
    # sequence-hidden layout and a maximum sequence length equal to the tensor length, using a
    # fake parallel state object that carries only the context parallel rank and size. Returns
    # the slice of the given rank.
    return slice_with_cp(
        tokens,
        pad_value=0,
        parallel_state=SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size)),
        qkv_format="bshd",
        max_seq_len=len(tokens),
    )


# ---------------------------------------------------------------------------
# TestBuildLabels
# ---------------------------------------------------------------------------


class TestBuildLabels:
    # [CLASS DOC] Tests for label construction: with context parallel size 1 labels are the inputs
    # shifted left by one with the ignore value minus 100 at the end; with larger sizes the label of
    # each local token is looked up from the global sequence using its global position id, and the
    # token at the last global position gets the ignore value.
    def test_cp1_next_token_shift(self) -> None:
        # [FUNCTION DOC] Inputs 10, 20, 30 give labels 20, 30 and the ignore value; checks shape
        # (1, 3) and long integer dtype.
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        position_ids = torch.arange(3).unsqueeze(0)
        global_input_ids = input_ids.clone()

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=1,
        )
        assert labels.shape == (1, 3)
        assert labels.dtype == torch.long
        assert labels[0].tolist() == [20, 30, -100]

    def test_cp1_batch_size_2(self) -> None:
        # [FUNCTION DOC] Two rows are shifted independently, each ending with the ignore value.
        input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.long)
        position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)
        global_input_ids = input_ids.clone()

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=1,
        )
        assert labels.shape == (2, 3)
        assert labels[0].tolist() == [20, 30, -100]
        assert labels[1].tolist() == [50, 60, -100]

    def test_cp1_single_token(self) -> None:
        # [FUNCTION DOC] A single token has no successor, so its only label is the ignore value.
        input_ids = torch.tensor([[99]], dtype=torch.long)
        position_ids = torch.tensor([[0]])
        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=input_ids.clone(),
            cp_size=1,
        )
        assert labels[0].tolist() == [-100]

    def test_cp_gt1_uses_position_ids(self) -> None:
        # [FUNCTION DOC] With context parallel size 2, local positions 0 and 3 of a global
        # sequence of four: position 0 gets the global token at position 1 (20) and position 3,
        # the last global position, gets the ignore value.
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[0, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels.shape == (1, 2)
        assert labels[0, 0].item() == 20  # next token after position 0
        assert labels[0, 1].item() == -100  # position 3 is last -> ignored

    def test_cp_gt1_middle_positions(self) -> None:
        # [FUNCTION DOC] Non-adjacent middle positions 1 and 4 of a six-token sequence get the
        # global tokens that follow them (30 and 60); no ignore value appears.
        global_input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        position_ids = torch.tensor([[1, 4]], dtype=torch.long)
        input_ids = torch.tensor([[20, 50]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0, 0].item() == 30  # next after pos 1
        assert labels[0, 1].item() == 60  # next after pos 4

    def test_cp_gt1_second_to_last_position_is_valid(self) -> None:
        # [FUNCTION DOC] Position 2 of four (second to last) is valid and gets token 40, while
        # position 3 (last) gets the ignore value.
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[2, 3]], dtype=torch.long)
        input_ids = torch.tensor([[30, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0, 0].item() == 40  # pos 2 -> next is pos 3 = token 40
        assert labels[0, 1].item() == -100  # pos 3 is last

    def test_cp_gt1_batch_size_2(self) -> None:
        # [FUNCTION DOC] Two rows with their own global sequences and the same position ids
        # (0 and 3) produce per-row labels: next token for position 0, ignore value for the last.
        global_input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long)
        position_ids = torch.tensor([[0, 3], [0, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 40], [50, 80]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels.shape == (2, 2)
        assert labels[0].tolist() == [20, -100]
        assert labels[1].tolist() == [60, -100]

    def test_cp_gt1_all_positions_valid(self) -> None:
        # [FUNCTION DOC] Positions 0, 2, 3 of a six-token sequence all have successors, so
        # labels are the tokens at positions 1, 3, 4 and nothing is ignored.
        global_input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        position_ids = torch.tensor([[0, 2, 3]], dtype=torch.long)
        input_ids = torch.tensor([[10, 30, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0].tolist() == [20, 40, 50]  # all valid, none is last pos

    def test_cp_gt1_all_positions_are_last(self) -> None:
        # [FUNCTION DOC] When every local position equals the last global position, every label
        # is the ignore value.
        global_input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        position_ids = torch.tensor([[3, 3]], dtype=torch.long)
        input_ids = torch.tensor([[40, 40]], dtype=torch.long)

        labels = _build_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            global_input_ids=global_input_ids,
            cp_size=2,
        )
        assert labels[0].tolist() == [-100, -100]


# ---------------------------------------------------------------------------
# TestLossFunc
# ---------------------------------------------------------------------------


class TestLossFunc:
    # [CLASS DOC] Tests for the loss function, which takes labels and model output logits of shape
    # (batch, sequence, vocabulary) and returns a pair of a scalar loss tensor and a dictionary of
    # metrics holding a detached copy of the loss under the key "loss". Positions labelled with the
    # ignore value minus 100 do not contribute.
    def test_return_type(self) -> None:
        # [FUNCTION DOC] Asserts the result is a tuple of length two: a tensor, then a
        # dictionary containing the key "loss".
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, -100]], dtype=torch.long)
        result = loss_func(labels=labels, output_tensor=logits)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], dict)
        assert "loss" in result[1]

    def test_loss_scalar(self) -> None:
        # [FUNCTION DOC] Both the loss and the metrics loss are zero-dimensional tensors.
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss, metrics = loss_func(labels=labels, output_tensor=logits)
        assert loss.dim() == 0
        assert metrics["loss"].dim() == 0

    def test_loss_finite(self) -> None:
        # [FUNCTION DOC] Random logits with valid labels give a finite loss.
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss, _ = loss_func(labels=labels, output_tensor=logits)
        assert torch.isfinite(loss)

    def test_loss_detached_in_metrics(self) -> None:
        # [FUNCTION DOC] Even when logits require gradients, the loss stored in the metrics
        # dictionary does not require gradients (it is detached).
        logits = torch.randn(1, 4, 10, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        _, metrics = loss_func(labels=labels, output_tensor=logits)
        assert not metrics["loss"].requires_grad

    def test_ignores_neg100(self) -> None:
        # [FUNCTION DOC] With all labels set to the ignore value the loss is either not-a-number
        # (empty mean) or exactly zero; both are accepted.
        logits = torch.randn(1, 4, 10)
        all_masked = torch.full((1, 4), -100, dtype=torch.long)
        loss, _ = loss_func(labels=all_masked, output_tensor=logits)
        assert torch.isnan(loss) or loss.item() == 0.0

    def test_perfect_prediction(self) -> None:
        # [FUNCTION DOC] Logits of 100 on the labelled class and minus 100 elsewhere give a loss
        # below 0.01.
        vocab_size = 5
        logits = torch.full((1, 3, vocab_size), -100.0)
        labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
        for pos in range(3):
            logits[0, pos, labels[0, pos]] = 100.0
        loss, _ = loss_func(labels=labels, output_tensor=logits)
        assert loss.item() < 0.01

    def test_batch_size_2(self) -> None:
        # [FUNCTION DOC] A batch of two sequences still yields a finite scalar loss.
        vocab_size = 10
        logits = torch.randn(2, 4, vocab_size)
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        loss, _ = loss_func(labels=labels, output_tensor=logits)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# TestPrepareBatch — cp_size=1
# ---------------------------------------------------------------------------


class TestPrepareBatchCP1:
    # [CLASS DOC] Tests for batch preparation without context parallel (default): the token list
    # is repeated along a new batch dimension, positions are sequential, labels are next tokens,
    # the attention mask is absent and the global ids equal the input ids.
    def test_all_keys_present(self) -> None:
        # [FUNCTION DOC] The batch has exactly the keys input ids, position ids, attention mask,
        # labels and global input ids.
        batch = prepare_batch(token_ids=list(range(8)), batch_size=1, device="cpu")
        expected_keys = {"input_ids", "position_ids", "attention_mask", "labels", "global_input_ids"}
        assert set(batch.keys()) == expected_keys

    def test_all_tensors_long_except_mask(self) -> None:
        # [FUNCTION DOC] Input ids, position ids, labels and global input ids are long integer
        # tensors; the attention mask is none.
        batch = prepare_batch(token_ids=list(range(8)), batch_size=1, device="cpu")
        assert batch["input_ids"].dtype == torch.long
        assert batch["position_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["global_input_ids"].dtype == torch.long
        assert batch["attention_mask"] is None

    def test_shapes(self) -> None:
        # [FUNCTION DOC] Eight tokens with batch size 2 give shape (2, 8) for every tensor.
        token_ids = list(range(8))
        batch = prepare_batch(token_ids=token_ids, batch_size=2, device="cpu")
        assert batch["input_ids"].shape == (2, 8)
        assert batch["position_ids"].shape == (2, 8)
        assert batch["attention_mask"] is None
        assert batch["labels"].shape == (2, 8)
        assert batch["global_input_ids"].shape == (2, 8)

    def test_input_ids_values(self) -> None:
        # [FUNCTION DOC] Input ids reproduce the given tokens in order.
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["input_ids"][0].tolist() == [10, 20, 30]

    def test_position_ids_sequential(self) -> None:
        # [FUNCTION DOC] Position ids count from 0 to length minus one.
        token_ids = list(range(5))
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["position_ids"][0].tolist() == [0, 1, 2, 3, 4]

    def test_attention_mask_is_none(self) -> None:
        # [FUNCTION DOC] The attention mask entry is none.
        token_ids = list(range(4))
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["attention_mask"] is None

    def test_labels_next_token(self) -> None:
        # [FUNCTION DOC] Labels are the next tokens, with the ignore value at the end.
        token_ids = [10, 20, 30, 40]
        batch = prepare_batch(token_ids=token_ids, batch_size=1, device="cpu")
        assert batch["labels"][0].tolist() == [20, 30, 40, -100]

    def test_global_input_ids_equals_input_ids(self) -> None:
        # [FUNCTION DOC] Without context parallel the global input ids tensor equals the input
        # ids tensor (batch size 2).
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=2, device="cpu")
        assert torch.equal(batch["global_input_ids"], batch["input_ids"])

    def test_batch_dim_broadcast(self) -> None:
        # [FUNCTION DOC] With batch size 3 every row holds the same tokens and the same
        # sequential positions.
        token_ids = [10, 20, 30]
        batch = prepare_batch(token_ids=token_ids, batch_size=3, device="cpu")
        for i in range(3):
            assert batch["input_ids"][i].tolist() == [10, 20, 30]
            assert batch["position_ids"][i].tolist() == [0, 1, 2]

    def test_single_token(self) -> None:
        # [FUNCTION DOC] One token yields input [42], label list of just the ignore value, and
        # no attention mask.
        batch = prepare_batch(token_ids=[42], batch_size=1, device="cpu")
        assert batch["input_ids"][0].tolist() == [42]
        assert batch["labels"][0].tolist() == [-100]
        assert batch["attention_mask"] is None


# ---------------------------------------------------------------------------
# TestPrepareBatch — cp_size > 1 (zigzag)
# ---------------------------------------------------------------------------


class TestPrepareBatchZigzag:
    # [CLASS DOC] Tests of batch preparation with context parallel size above 1, exercising the
    # real zigzag slicing routine: the sequence is cut into twice cp_size chunks and rank r takes
    # chunk r plus chunk (2 * cp_size - 1 - r). Fixtures prepare the batches of all ranks.
    """Tests that exercise the real slice_with_cp zigzag CP path."""

    @pytest.fixture()
    def seq8_cp2(self) -> dict[str, dict[str, torch.Tensor]]:
        # [FUNCTION DOC] Fixture returning, for sequence tokens 100 to 800 (step 100) of length 8
        # and context parallel size 2, a mapping from "rank0"/"rank1" to the batch (batch size 1,
        # CPU) prepared for that rank.
        """Prepare batches for all ranks with seq_len=8, cp_size=2.

        Zigzag with cp_size=2, seq_len=8:
          chunk_size = ceil(8 / 4) = 2
          rank 0: tokens[0:2] + tokens[6:8] = positions [0,1,6,7]
          rank 1: tokens[2:4] + tokens[4:6] = positions [2,3,4,5]
        """
        token_ids = [100, 200, 300, 400, 500, 600, 700, 800]
        batches: dict[str, dict[str, torch.Tensor]] = {}
        for rank in range(2):
            batches[f"rank{rank}"] = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=2,
                device="cpu",
            )
        return batches

    @pytest.fixture()
    def seq16_cp4(self) -> dict[str, dict[str, torch.Tensor]]:
        # [FUNCTION DOC] Fixture returning, for tokens 1000 to 1015 and context parallel size 4,
        # a mapping from "rank0" to "rank3" to the batch (batch size 1, CPU) of that rank.
        """Prepare batches for all ranks with seq_len=16, cp_size=4.

        chunk_size = ceil(16 / 8) = 2
        rank 0: tokens[0:2] + tokens[14:16]
        rank 1: tokens[2:4] + tokens[12:14]
        rank 2: tokens[4:6] + tokens[10:12]
        rank 3: tokens[6:8] + tokens[8:10]
        """
        token_ids = list(range(1000, 1016))
        batches: dict[str, dict[str, torch.Tensor]] = {}
        for rank in range(4):
            batches[f"rank{rank}"] = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=4,
                device="cpu",
            )
        return batches

    def test_zigzag_token_and_position_assignment_cp2(self, seq8_cp2: dict) -> None:
        # [FUNCTION DOC] Checks, for each of the two ranks, the exact local token ids and global
        # positions (rank 0: positions 0, 1, 6, 7; rank 1: positions 2 to 5), local shapes of
        # (1, 4) for inputs and labels, no attention mask, and that the global input ids keep the
        # full length-8 sequence.
        """Verify exact zigzag token/position assignment for all ranks with cp_size=2.

        seq_len=8, chunk_size=2:
          rank 0: tokens[0:2] + tokens[6:8] -> ids [100,200,700,800], pos [0,1,6,7]
          rank 1: tokens[2:4] + tokens[4:6] -> ids [300,400,500,600], pos [2,3,4,5]
        """
        expected_ids = {
            "rank0": [100, 200, 700, 800],
            "rank1": [300, 400, 500, 600],
        }
        expected_pos = {
            "rank0": [0, 1, 6, 7],
            "rank1": [2, 3, 4, 5],
        }
        for rank_key in seq8_cp2:
            batch = seq8_cp2[rank_key]
            assert batch["input_ids"][0].tolist() == expected_ids[rank_key]
            assert batch["position_ids"][0].tolist() == expected_pos[rank_key]

            assert batch["input_ids"].shape == (1, 4)
            assert batch["labels"].shape == (1, 4)
            assert batch["attention_mask"] is None
            assert batch["global_input_ids"].shape == (1, 8)
            assert batch["global_input_ids"][0].tolist() == [100, 200, 300, 400, 500, 600, 700, 800]

    def test_zigzag_token_and_position_assignment_cp4(self, seq16_cp4: dict) -> None:
        # [FUNCTION DOC] Same exact-assignment check for four ranks and a length-16 sequence:
        # each rank holds two chunks of two tokens (local shape (1, 4)), and the global input
        # ids keep length 16.
        """Verify exact zigzag token/position assignment for all ranks with cp_size=4.

        seq_len=16, chunk_size=2:
          rank 0: tokens[0:2]  + tokens[14:16] -> [1000,1001,1014,1015], pos [0,1,14,15]
          rank 1: tokens[2:4]  + tokens[12:14] -> [1002,1003,1012,1013], pos [2,3,12,13]
          rank 2: tokens[4:6]  + tokens[10:12] -> [1004,1005,1010,1011], pos [4,5,10,11]
          rank 3: tokens[6:8]  + tokens[8:10]  -> [1006,1007,1008,1009], pos [6,7,8,9]
        """
        expected_ids = {
            "rank0": [1000, 1001, 1014, 1015],
            "rank1": [1002, 1003, 1012, 1013],
            "rank2": [1004, 1005, 1010, 1011],
            "rank3": [1006, 1007, 1008, 1009],
        }
        expected_pos = {
            "rank0": [0, 1, 14, 15],
            "rank1": [2, 3, 12, 13],
            "rank2": [4, 5, 10, 11],
            "rank3": [6, 7, 8, 9],
        }
        for rank_key in seq16_cp4:
            batch = seq16_cp4[rank_key]
            assert batch["input_ids"][0].tolist() == expected_ids[rank_key]
            assert batch["position_ids"][0].tolist() == expected_pos[rank_key]

            assert batch["input_ids"].shape == (1, 4)
            assert batch["labels"].shape == (1, 4)
            assert batch["global_input_ids"].shape == (1, 16)

    def test_all_ranks_partition_full_sequence(self, seq8_cp2: dict, seq16_cp4: dict) -> None:
        # [FUNCTION DOC] For both fixtures, the union of positions over all ranks is exactly
        # the range of the sequence length with no duplicates, and the total token count equals
        # the sequence length.
        """All ranks together cover every position exactly once (no overlap, no gap)."""
        for batches, seq_len in [(seq8_cp2, 8), (seq16_cp4, 16)]:
            all_positions: list[int] = []
            all_tokens: list[int] = []
            for rank_key in batches:
                all_positions.extend(batches[rank_key]["position_ids"][0].tolist())
                all_tokens.extend(batches[rank_key]["input_ids"][0].tolist())

            assert sorted(all_positions) == list(range(seq_len))
            assert len(all_positions) == len(set(all_positions))
            assert len(all_tokens) == seq_len

    def test_labels_per_rank_cp2(self, seq8_cp2: dict) -> None:
        # [FUNCTION DOC] Checks exact labels per rank for context parallel size 2 (rank 0 ends
        # with the ignore value for global position 7) and that exactly one ignore value exists
        # over all ranks.
        """Verify labels for all ranks with cp_size=2.

        rank 0 positions [0,1,6,7]: labels = [tok@1=200, tok@2=300, tok@7=800, -100]
        rank 1 positions [2,3,4,5]: labels = [tok@3=400, tok@4=500, tok@5=600, tok@6=700]
        """
        expected_labels = {
            "rank0": [200, 300, 800, -100],
            "rank1": [400, 500, 600, 700],
        }
        for rank_key in seq8_cp2:
            assert seq8_cp2[rank_key]["labels"][0].tolist() == expected_labels[rank_key]

        neg100_count = sum((seq8_cp2[rk]["labels"] == -100).sum().item() for rk in seq8_cp2)
        assert neg100_count == 1

    def test_labels_per_rank_cp4(self, seq16_cp4: dict) -> None:
        # [FUNCTION DOC] Same label check for four ranks: only rank 0 (holding the last global
        # position 15) has the ignore value, and exactly one exists in total.
        """Verify labels for all ranks with cp_size=4.

        rank 0 positions [0,1,14,15]:  labels = [1001, 1002, 1015, -100]
        rank 1 positions [2,3,12,13]:  labels = [1003, 1004, 1013, 1014]
        rank 2 positions [4,5,10,11]:  labels = [1005, 1006, 1011, 1012]
        rank 3 positions [6,7,8,9]:    labels = [1007, 1008, 1009, 1010]
        """
        expected_labels = {
            "rank0": [1001, 1002, 1015, -100],
            "rank1": [1003, 1004, 1013, 1014],
            "rank2": [1005, 1006, 1011, 1012],
            "rank3": [1007, 1008, 1009, 1010],
        }
        for rank_key in seq16_cp4:
            assert seq16_cp4[rank_key]["labels"][0].tolist() == expected_labels[rank_key]

        neg100_count = sum((seq16_cp4[rk]["labels"] == -100).sum().item() for rk in seq16_cp4)
        assert neg100_count == 1

    def test_input_ids_match_global_at_positions(self, seq8_cp2: dict, seq16_cp4: dict) -> None:
        # [FUNCTION DOC] For every rank in both fixtures, gathering the global input ids at the
        # local position ids reproduces the local input ids.
        """For every rank, input_ids[i] == global_input_ids[position_ids[i]]."""
        for batches in [seq8_cp2, seq16_cp4]:
            for rank_key in batches:
                batch = batches[rank_key]
                gathered = batch["global_input_ids"][0][batch["position_ids"][0]]
                assert torch.equal(batch["input_ids"][0], gathered)

    def test_attention_mask_is_none_zigzag(self, seq8_cp2: dict) -> None:
        # [FUNCTION DOC] The attention mask stays none on every rank under zigzag slicing.
        for rank_key in seq8_cp2:
            assert seq8_cp2[rank_key]["attention_mask"] is None

    def test_positions_monotonic_within_each_chunk(self, seq8_cp2: dict, seq16_cp4: dict) -> None:
        # [FUNCTION DOC] Splits each rank's local positions into a first and second half and
        # asserts each half is strictly increasing (the halves themselves are not compared).
        """Each half (chunk) of local positions should be monotonically increasing."""
        for batches in [seq8_cp2, seq16_cp4]:
            for rank_key in batches:
                pos = batches[rank_key]["position_ids"][0]
                half = len(pos) // 2
                chunk1, chunk2 = pos[:half], pos[half:]
                assert all(chunk1[i] < chunk1[i + 1] for i in range(len(chunk1) - 1))
                assert all(chunk2[i] < chunk2[i + 1] for i in range(len(chunk2) - 1))

    def test_batch_dim_broadcast_cp2(self) -> None:
        # [FUNCTION DOC] With batch size 3, rank 0 and context parallel size 2, the first
        # dimension is 3 and all rows have identical input ids and labels.
        token_ids = list(range(100, 108))
        batch = prepare_batch(token_ids=token_ids, batch_size=3, cp_rank=0, cp_size=2, device="cpu")
        assert batch["input_ids"].shape[0] == 3
        for i in range(3):
            assert batch["input_ids"][i].tolist() == batch["input_ids"][0].tolist()
            assert batch["labels"][i].tolist() == batch["labels"][0].tolist()


# ---------------------------------------------------------------------------
# TestPrepareBatch — zigzag with padding (seq_len not divisible by 2*cp_size)
# ---------------------------------------------------------------------------


class TestPrepareBatchZigzagPadding:
    # [CLASS DOC] Tests for sequences whose length is not divisible by twice the context parallel
    # size: the sequence is padded with token 0 up to a multiple before slicing, so each rank still
    # gets an equal local length.
    """When seq_len is not evenly divisible by 2*cp_size, slice_with_cp pads."""

    def test_seq7_cp2_shapes(self) -> None:
        # [FUNCTION DOC] Seven tokens with size 2 give a local input of length 4 on rank 0 while
        # the global input ids keep the original length 7 (unpadded).
        """seq_len=7, cp_size=2 -> chunk_size=ceil(7/4)=2, padded_len=8, local=4."""
        token_ids = list(range(10, 17))  # 7 tokens
        batch = prepare_batch(token_ids=token_ids, batch_size=1, cp_rank=0, cp_size=2, device="cpu")
        assert batch["input_ids"].shape == (1, 4)
        assert batch["global_input_ids"].shape == (1, 7)

    def test_seq7_cp2_padded_token_is_zero(self) -> None:
        # [FUNCTION DOC] On rank 0 the local ids are 10, 11, 16 and then a padding token 0 (the
        # padded global slot 7).
        """The padding token (pad_value=0) should appear in the local slice."""
        token_ids = list(range(10, 17))  # 7 tokens
        batch_r0 = prepare_batch(token_ids=token_ids, batch_size=1, cp_rank=0, cp_size=2, device="cpu")
        # rank 0: tokens[0:2] + tokens[6:8] where tokens[7]=0 (padded)
        local_ids = batch_r0["input_ids"][0].tolist()
        assert local_ids[0] == 10
        assert local_ids[1] == 11
        assert local_ids[2] == 16  # original token at index 6
        assert local_ids[3] == 0  # padded

    def test_seq7_cp2_all_ranks_cover_original_tokens(self) -> None:
        # [FUNCTION DOC] Collects (position, token) pairs over both ranks and asserts the set of
        # positions below 7 equals the full original range, i.e. no real token is lost to
        # padding.
        token_ids = list(range(10, 17))
        all_token_pos: list[tuple[int, int]] = []
        for rank in range(2):
            batch = prepare_batch(token_ids=token_ids, batch_size=1, cp_rank=rank, cp_size=2, device="cpu")
            pos = batch["position_ids"][0].tolist()
            ids = batch["input_ids"][0].tolist()
            all_token_pos.extend(zip(pos, ids, strict=True))

        original_positions = {p for p, _ in all_token_pos if p < 7}
        assert original_positions == set(range(7))

    def test_seq5_cp2_shapes(self) -> None:
        # [FUNCTION DOC] Five tokens with size 2 are padded to 8, so the local input has length 4.
        """seq_len=5, cp_size=2 -> chunk_size=ceil(5/4)=2, padded=8, local=4."""
        token_ids = list(range(5))
        batch = prepare_batch(token_ids=token_ids, batch_size=1, cp_rank=0, cp_size=2, device="cpu")
        assert batch["input_ids"].shape == (1, 4)

    @pytest.mark.parametrize("seq_len", [3, 5, 7, 9, 11, 13])
    def test_odd_seq_len_cp2_no_crash(self, seq_len: int) -> None:
        # [FUNCTION DOC] Parametrized over odd lengths 3 to 13 with size 2: preparing batches
        # for both ranks must not fail, with batch dimension 1, matching input and position
        # lengths, and labels the same shape as inputs.
        token_ids = list(range(seq_len))
        for rank in range(2):
            batch = prepare_batch(token_ids=token_ids, batch_size=1, cp_rank=rank, cp_size=2, device="cpu")
            assert batch["input_ids"].shape[0] == 1
            assert batch["input_ids"].shape[1] == batch["position_ids"].shape[1]
            assert batch["labels"].shape == batch["input_ids"].shape

    @pytest.mark.parametrize("seq_len", [5, 7, 10, 13, 15])
    def test_various_seq_len_cp4_no_crash(self, seq_len: int) -> None:
        # [FUNCTION DOC] Parametrized over lengths 5, 7, 10, 13, 15 with size 4: preparing
        # batches for all four ranks must not fail, with batch dimension 1 and labels the same
        # shape as inputs.
        token_ids = list(range(seq_len))
        for rank in range(4):
            batch = prepare_batch(token_ids=token_ids, batch_size=1, cp_rank=rank, cp_size=4, device="cpu")
            assert batch["input_ids"].shape[0] == 1
            assert batch["labels"].shape == batch["input_ids"].shape


# ---------------------------------------------------------------------------
# TestPrepareBatch — label cross-validation across all ranks
# ---------------------------------------------------------------------------


class TestLabelsCrossRankConsistency:
    # [CLASS DOC] Tests that labels assembled from all context parallel ranks match naive
    # next-token targets of the full sequence.
    """Verify that labels across all CP ranks, when reassembled, form correct next-token targets."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_reassembled_labels_match_naive(self, cp_size: int) -> None:
        # [FUNCTION DOC] For sequence length four times the size (evenly divisible), builds a
        # mapping from global position to label across all ranks and asserts position p has the
        # token at p+1 and the last position has the ignore value.
        """Gather (position, label) from all ranks and verify against naive next-token."""
        seq_len = cp_size * 4  # evenly divisible
        token_ids = list(range(1000, 1000 + seq_len))

        position_to_label: dict[int, int] = {}
        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            positions = batch["position_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            for pos, label in zip(positions, labels, strict=True):
                position_to_label[pos] = label

        for pos in range(seq_len - 1):
            assert (
                position_to_label[pos] == token_ids[pos + 1]
            ), f"At pos={pos}, expected label={token_ids[pos + 1]}, got {position_to_label[pos]}"
        assert position_to_label[seq_len - 1] == -100

    @pytest.mark.parametrize("cp_size,seq_len", [(2, 7), (2, 11), (4, 9), (4, 15)])
    def test_reassembled_labels_match_naive_with_padding(self, cp_size: int, seq_len: int) -> None:
        # [FUNCTION DOC] Same cross-rank check with non-divisible lengths: entries from padded
        # positions (at or beyond the real length) are discarded, and every real position except
        # the last must have the next token as label. The last real position is not asserted.
        """Same cross-rank check but with non-divisible seq_len (padding involved)."""
        token_ids = list(range(1000, 1000 + seq_len))

        position_to_label: dict[int, int] = {}
        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            positions = batch["position_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            for pos, label in zip(positions, labels, strict=True):
                if pos < seq_len:
                    position_to_label[pos] = label

        for pos in range(seq_len - 1):
            assert (
                position_to_label.get(pos) == token_ids[pos + 1]
            ), f"At pos={pos}, expected label={token_ids[pos + 1]}, got {position_to_label.get(pos)}"


# ---------------------------------------------------------------------------
# TestPrepareBatch — slice_with_cp consistency
# ---------------------------------------------------------------------------


class TestSliceWithCPConsistency:
    # [CLASS DOC] Tests that the batch preparation slices inputs and positions exactly like a
    # direct call of the production slicing routine (via the module's reference helper).
    """Verify that prepare_batch's CP slicing matches direct slice_with_cp calls."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_input_ids_match_direct_slice(self, cp_size: int) -> None:
        # [FUNCTION DOC] For every rank, the first row of the prepared input ids equals the
        # direct slice of the token tensor (length four times the size).
        seq_len = cp_size * 4
        token_ids = list(range(1000, 1000 + seq_len))
        tokens_tensor = torch.tensor(token_ids, dtype=torch.long)

        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            expected = _zigzag_slice(tokens_tensor, cp_rank=rank, cp_size=cp_size)
            assert torch.equal(batch["input_ids"][0], expected)

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_position_ids_match_direct_slice(self, cp_size: int) -> None:
        # [FUNCTION DOC] For every rank, the first row of the prepared position ids equals the
        # direct slice of the range of global positions.
        seq_len = cp_size * 4
        token_ids = list(range(seq_len))
        positions_tensor = torch.arange(seq_len, dtype=torch.long)

        for rank in range(cp_size):
            batch = prepare_batch(
                token_ids=token_ids,
                batch_size=1,
                cp_rank=rank,
                cp_size=cp_size,
                device="cpu",
            )
            expected = _zigzag_slice(positions_tensor, cp_rank=rank, cp_size=cp_size)
            assert torch.equal(batch["position_ids"][0], expected)
