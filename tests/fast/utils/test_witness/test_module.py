# [MODULE DOC] Tests for the data witness module: a tiny embedding-backed layer whose forward
# output leaves hidden states bitwise unchanged but whose backward leaves gradient on the rows
# addressed by witness ids (so optimizer updates reveal which data ids were seen); installation
# into a (fake) GPT model as head and tail submodules; recording nonzero rows through the event
# logger; zeroing stale rows in distributed optimizer shards and Adam state; and the absolute
# value broadcast add autograd function (forward is plain addition; the addend gradient is the
# absolute-value sum over the last dim so mixed-sign gradients do not cancel). Hidden states use
# the sequence-first layout (sequence, batch, hidden). Megatron optimizer classes are built
# without running their constructors, as minimal fakes.
"""Tests for miles.utils.audit_utils.witness.module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer import ChainedOptimizer

from miles.utils.audit_utils.event_logger.models import WitnessSnapshotParamEvent
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.audit_utils.witness.module import (
    _abs_broadcast_add,
    _AbsBroadcastAdd,
    _DataWitness,
    _record_and_log_witness_param,
    _zero_witness_rows,
    install_witness,
    witness_dump_and_clear_stale,
)


class TestDataWitnessForward:
    # [CLASS DOC] Tests of the witness layer's forward and backward behavior.
    def test_forward_does_not_change_hidden_states(self) -> None:
        # [FUNCTION DOC] A fresh witness with ids and random hidden states returns hidden states
        # exactly equal to the input.
        """Witness output is zero, so hidden_states should be unchanged."""
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[0, 1, 2, 3]])  # [1, 4]
        hidden = torch.randn(4, 1, 8)  # [s, b, h] Megatron SBH layout
        result = witness(ids, hidden)
        assert torch.equal(result, hidden)

    def test_forward_unchanged_after_optimizer_step(self) -> None:
        # [FUNCTION DOC] Runs a forward, backward and an Adam step so the witness weights become
        # nonzero; asserts the weights are not all zero and a second forward still returns hidden
        # states exactly equal to the input.
        witness = _DataWitness(buffer_size=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        ids = torch.tensor([[0, 1, 2]])
        hidden = torch.randn(3, 1, 8)
        result = witness(ids, hidden)
        result.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        # After optimizer update, weights are nonzero, but hidden_states still unchanged
        assert not torch.all(witness.witness.weight == 0.0)
        result2 = witness(ids, hidden)
        assert torch.equal(result2, hidden)

    def test_backward_records_gradient_on_witness_weight(self) -> None:
        # [FUNCTION DOC] With witness ids 2 and 5, backward of the output sum leaves nonzero
        # gradient on exactly rows 2 and 5 of the witness weight.
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[2, 5]])
        hidden = torch.randn(2, 1, 4, requires_grad=True)

        result = witness(ids, hidden)
        result.sum().backward()

        grad = witness.witness.weight.grad
        assert grad is not None
        nonzero_rows = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()
        assert set(nonzero_rows) == {2, 5}

    def test_no_effect_on_main_model_gradients(self) -> None:
        # [FUNCTION DOC] Seeded; computes gradients of an embedding and a linear layer without a
        # witness, then again with the witness inserted between them (ids all zero); asserts both
        # parameter gradients are exactly equal.
        """Witness should not alter gradients for upstream or downstream model parameters."""
        torch.manual_seed(42)
        embed = nn.Embedding(100, 8)
        linear = nn.Linear(8, 1)

        tokens = torch.tensor([[1, 2, 3, 4]])

        # Step 1: Compute loss without witness
        hidden = embed(tokens).transpose(0, 1).contiguous()  # [s=4, b=1, h=8]
        out_no_witness = linear(hidden).sum()
        out_no_witness.backward()
        grad_embed_no = embed.weight.grad.clone()
        grad_linear_no = linear.weight.grad.clone()

        embed.zero_grad()
        linear.zero_grad()

        # Step 2: Compute loss with witness
        hidden = embed(tokens).transpose(0, 1).contiguous()
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[0, 0, 0, 0]])
        h = witness(ids, hidden)
        out_with_witness = linear(h).sum()
        out_with_witness.backward()

        assert torch.equal(grad_embed_no, embed.weight.grad)
        assert torch.equal(grad_linear_no, linear.weight.grad)


class TestRecordAndLogWitnessParam:
    # [CLASS DOC] Tests of the function that logs a witness snapshot event; the event logger
    # getter is patched with a mock logger.
    def test_logs_nonzero_weight_rows(self) -> None:
        # [FUNCTION DOC] Rows 3 and 7 are set nonzero. One log call is made whose payload (second
        # argument) lists nonzero witness ids 3 and 7 and the given instance id.
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[3] = 1.0
        witness.witness.weight.data[7] = 2.0

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.head", stale_ids=[])

            mock_logger.log.assert_called_once()
            # New API: log(event_cls, partial_dict)
            partial = mock_logger.log.call_args[0][1]
            assert set(partial["nonzero_witness_ids"]) == {3, 7}
            assert partial["instance_id"] == "pp0.head"

    def test_record_and_log_event_fields(self) -> None:
        # [FUNCTION DOC] Rows 1 and 4 nonzero; one log call whose first argument is the witness
        # snapshot event class and whose payload has the instance id and nonzero ids 1 and 4.
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[1] = 0.5
        witness.witness.weight.data[4] = -0.3

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.tail", stale_ids=[])

            mock_logger.log.assert_called_once()
            assert mock_logger.log.call_args[0][0] is WitnessSnapshotParamEvent
            partial = mock_logger.log.call_args[0][1]
            assert partial["instance_id"] == "pp0.tail"
            assert set(partial["nonzero_witness_ids"]) == {1, 4}


# ---------------------------------------------------------------------------
# Fake GPTModel for install_witness / forward integration tests
# ---------------------------------------------------------------------------


class _FakeDecoder(nn.Module):
    # [CLASS DOC] HELPER: Fake decoder that holds an optional input tensor (as pipeline middle
    # stages receive) and whose forward returns the hidden states unchanged.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the input tensor slot to none.
        super().__init__()
        self.input_tensor: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns the hidden states; extra keyword arguments are ignored.
        return hidden_states


class _FakeGPTModel(nn.Module):
    # [CLASS DOC] HELPER: Fake GPT model with a pre-process flag, a fake decoder and a 100 by 16
    # embedding, mimicking where install_witness hooks in.
    def __init__(self, *, pre_process: bool = True) -> None:
        # [FUNCTION DOC] HELPER: Stores the flag and builds the decoder and embedding.
        super().__init__()
        self.pre_process = pre_process
        self.decoder = _FakeDecoder()
        self.embedding = nn.Embedding(100, 16)

    def forward(self, input_ids: torch.Tensor, witness_ids: torch.Tensor | None = None) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: On the first pipeline stage embeds the input ids into a
        # sequence-first tensor; otherwise there is no decoder input. If a head witness was
        # installed and witness ids are given, passes the decoder input (or, on middle stages, the
        # decoder's stored input tensor, which is replaced by the result) through it. Falls back
        # to the stored input tensor when there is no decoder input, and returns the decoder's
        # output.
        if self.pre_process:
            # Megatron decoders use sequence-first [s, b, h] layout, which is what
            # _DataWitness expects (it transposes its own output to [s, b, 1]).
            decoder_input = self.embedding(input_ids).transpose(0, 1).contiguous()
        else:
            decoder_input = None

        if hasattr(self, "local_head_witness") and witness_ids is not None:
            if decoder_input is not None:
                decoder_input = self.local_head_witness(witness_ids, decoder_input)
            else:
                self.decoder.input_tensor = self.local_head_witness(witness_ids, self.decoder.input_tensor)

        if decoder_input is None:
            decoder_input = self.decoder.input_tensor

        return self.decoder(hidden_states=decoder_input)


class TestInstallWitness:
    # [CLASS DOC] Tests of installing head and tail witness submodules into the fake model.
    def test_witness_is_submodule(self) -> None:
        # [FUNCTION DOC] After installation both the head and tail witness appear as named
        # submodules.
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        assert "local_head_witness" in dict(model.named_modules())
        assert "local_tail_witness" in dict(model.named_modules())

    def test_witness_in_parameters(self) -> None:
        # [FUNCTION DOC] Both witnesses contribute named parameters.
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        param_names = [name for name, _ in model.named_parameters()]
        assert any("local_head_witness" in name for name in param_names)
        assert any("local_tail_witness" in name for name in param_names)

    def test_forward_adds_zero(self) -> None:
        # [FUNCTION DOC] Model output with witness ids equals the output without them.
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out_no = model(tokens)
        out_with = model(tokens, witness_ids=torch.tensor([[0, 1, 2]]))
        assert torch.equal(out_no, out_with)

    def test_forward_produces_gradient(self) -> None:
        # [FUNCTION DOC] With witness ids all 5, backward leaves gradient at row 5 of the head
        # witness weight.
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out = model(tokens, witness_ids=torch.tensor([[5, 5, 5]]))
        out.sum().backward()
        grad = model.local_head_witness.witness.weight.grad
        assert grad is not None
        assert 5 in grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_no_witness_ids_no_effect(self) -> None:
        # [FUNCTION DOC] A forward without witness ids still returns a result (not none).
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        out = model(torch.tensor([[1, 2, 3]]))
        assert out is not None

    def test_witness_in_state_dict(self) -> None:
        # [FUNCTION DOC] The state dict has keys for both witnesses.
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        sd = model.state_dict()
        assert any("local_head_witness" in k for k in sd)
        assert any("local_tail_witness" in k for k in sd)

    def test_checkpoint_roundtrip(self) -> None:
        # [FUNCTION DOC] Fills the head witness weight with 42, loads the state dict into a second
        # installed model, and asserts the weights are equal.
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        model.local_head_witness.witness.weight.data.fill_(42.0)
        sd = model.state_dict()

        model2 = _FakeGPTModel()
        install_witness(model2, buffer_size=10)
        model2.load_state_dict(sd)
        assert torch.equal(model2.local_head_witness.witness.weight.data, model.local_head_witness.witness.weight.data)

    def test_disabled_no_submodule(self) -> None:
        # [FUNCTION DOC] Without installation the model has no head witness attribute.
        model = _FakeGPTModel()
        assert not hasattr(model, "local_head_witness")

    def test_middle_pp_stage_modifies_input_tensor(self) -> None:
        # [FUNCTION DOC] On a model without pre-processing (a middle pipeline stage), the decoder
        # input tensor is given; the output equals the unchanged hidden states.
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        # Sequence-first: [s=4, b=1, h=16].
        hidden = torch.randn(4, 1, 16)
        model.decoder.input_tensor = hidden.clone()
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[0, 1, 2, 3]]))
        assert torch.equal(out, hidden)

    def test_middle_pp_stage_produces_gradient(self) -> None:
        # [FUNCTION DOC] On a middle stage with a gradient-requiring input tensor and witness ids
        # all 5, backward leaves gradient at row 5 of the head witness weight.
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        # Sequence-first: [s=4, b=1, h=16].
        model.decoder.input_tensor = torch.randn(4, 1, 16, requires_grad=True)
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[5, 5, 5, 5]]))
        out.sum().backward()
        assert 5 in model.local_head_witness.witness.weight.grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_forward_bitwise_zero_bf16(self) -> None:
        # [FUNCTION DOC] A witness cast to bfloat16 with all weights 3.14 still returns bfloat16
        # hidden states exactly equal to the input.
        witness = _DataWitness(buffer_size=10).to(dtype=torch.bfloat16)
        witness.witness.weight.data.fill_(3.14)
        ids = torch.tensor([[0, 1, 2]])
        hidden = torch.randn(3, 1, 8, dtype=torch.bfloat16)
        result = witness(ids, hidden)
        assert torch.equal(result, hidden)


class TestZeroWitnessRows:
    # [CLASS DOC] Tests of zeroing given witness rows in the weight and in the owning distributed
    # optimizer's float32 main shard and Adam moments, for optimizers built by the helper below.
    def test_zero_witness_rows_clears_distributed_optimizer_shard_rows(self) -> None:
        # [FUNCTION DOC] Weights all 1; the optimizer owns rows 4 to 10. Zeroing rows 2, 5, 7:
        # model weight rows 2, 5, 7 become 0 and row 4 stays 1; in the shard and both Adam
        # moments, local positions 1 and 3 (rows 5 and 7) are 0 and position 0 stays 1 (row 2 is
        # outside the shard).
        """Stale rows inside the local dist-opt shard are zeroed in the fp32 shard and its Adam state."""
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data.fill_(1.0)
        dist_opt, main_shard = _make_distributed_optimizer(witness.witness.weight, start=4, end=10)
        optimizer = ChainedOptimizer([dist_opt])

        _zero_witness_rows(witness=witness, idx=torch.tensor([2, 5, 7]), optimizer=optimizer)

        weight = witness.witness.weight
        for stale_row in (2, 5, 7):
            assert weight.data[stale_row].item() == 0.0
        assert weight.data[4].item() == 1.0
        # The shard covers rows [4, 10): row 5 -> local 1, row 7 -> local 3; row 2 is outside.
        state = dist_opt.optimizer.state[main_shard]
        for tensor in (main_shard, state["exp_avg"], state["exp_avg_sq"]):
            assert tensor[1].item() == 0.0
            assert tensor[3].item() == 0.0
            assert tensor[0].item() == 1.0

    def test_zero_witness_rows_skips_distributed_optimizer_without_the_param(self) -> None:
        # [FUNCTION DOC] A chained optimizer of the owning instance plus a second instance with no
        # buffer mapping for the parameter: zeroing row 3 clears it in the weight and the owner's
        # shard without error.
        """A chained dist-opt instance owning no shard of the witness param is skipped, the owner still clears."""
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data.fill_(1.0)
        dist_opt, main_shard = _make_distributed_optimizer(witness.witness.weight, start=0, end=10)
        expert_opt = DistributedOptimizer.__new__(DistributedOptimizer)
        expert_opt.config = dist_opt.config
        expert_opt.model_param_gbuf_map = {}
        optimizer = ChainedOptimizer([dist_opt, expert_opt])

        _zero_witness_rows(witness=witness, idx=torch.tensor([3]), optimizer=optimizer)

        assert witness.witness.weight.data[3].item() == 0.0
        assert main_shard[3].item() == 0.0

    def test_zero_witness_rows_rejects_non_distributed_optimizer(self) -> None:
        # [FUNCTION DOC] A plain Adam optimizer makes zeroing raise an assertion error mentioning
        # an unsupported optimizer.
        """Any optimizer other than (chained) DistributedOptimizer is rejected outright."""
        witness = _DataWitness(buffer_size=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.01)

        with pytest.raises(AssertionError, match="unsupported optimizer"):
            _zero_witness_rows(witness=witness, idx=torch.tensor([1]), optimizer=optimizer)

    def test_zero_witness_rows_raises_when_populated_state_misses_witness_shard(self) -> None:
        # [FUNCTION DOC] The inner optimizer state is replaced with an entry for an unrelated
        # tensor; zeroing raises an assertion error saying the witness main shard is missing.
        """A populated optimizer state without the witness shard entry fails loudly instead of silently skipping."""
        witness = _DataWitness(buffer_size=10)
        dist_opt, _main_shard = _make_distributed_optimizer(witness.witness.weight, start=0, end=10)
        unrelated = torch.ones(4)
        dist_opt.optimizer.state = {unrelated: {"exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)}}

        with pytest.raises(AssertionError, match="witness main shard missing"):
            _zero_witness_rows(witness=witness, idx=torch.tensor([1]), optimizer=ChainedOptimizer([dist_opt]))


# ---------------------------------------------------------------------------
# Helpers for witness_dump_and_clear_stale tests
# ---------------------------------------------------------------------------


def _make_fake_chunk(buffer_size: int = 10) -> nn.Module:
    # [FUNCTION DOC] HELPER: Builds a chunk module wrapping an inner module that has a head and a
    # tail data witness of the given buffer size.
    """Create a fake model chunk with .module.local_head_witness and .module.local_tail_witness."""
    inner = nn.Module()
    inner.local_head_witness = _DataWitness(buffer_size=buffer_size)
    inner.local_tail_witness = _DataWitness(buffer_size=buffer_size)
    chunk = nn.Module()
    chunk.module = inner
    return chunk


def _make_distributed_optimizer(
    model_weight: nn.Parameter,
    *,
    start: int,
    end: int,
) -> tuple[DistributedOptimizer, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Creates a distributed optimizer object without running its
    # constructor, filling only the attributes the code under test reads: an inner Adam over a
    # float32 copy (the main shard) of the flattened weight slice from start to end, with Adam
    # state of ones for both moments; a config with precision-aware and CPU offload off and the
    # adam type; and buffer, range and group index maps placing the weight at range start to end.
    # Returns the optimizer and the main shard tensor.
    """Build a minimal fake DistributedOptimizer owning rows [start, end) of model_weight."""
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    main_shard = model_weight.detach().view(-1)[start:end].clone().float().requires_grad_()
    inner = torch.optim.Adam([main_shard], lr=0.01)
    inner.state[main_shard] = {
        "exp_avg": torch.ones(end - start),
        "exp_avg_sq": torch.ones(end - start),
    }
    optimizer.optimizer = inner
    optimizer.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        optimizer_cpu_offload=False,
        optimizer="adam",
    )
    optimizer.model_param_gbuf_map = {model_weight: (0, torch.bfloat16, 0)}
    optimizer.gbuf_ranges = [{torch.bfloat16: [{"param_map": {model_weight: {"param": Range(start, end)}}}]}]
    optimizer.model_param_group_index_map = {model_weight: (0, 0)}
    return optimizer, main_shard


def _make_chained_optimizer_for_witnesses(model: list[nn.Module]) -> ChainedOptimizer:
    # [FUNCTION DOC] HELPER: Collects the head and tail witness weights of each chunk in order and
    # chains one fake distributed optimizer per weight, each owning the whole weight.
    """Build a fake ChainedOptimizer with one fully-owning dist-opt instance per witness weight."""
    weights = [
        getattr(chunk.module, attr).witness.weight
        for chunk in model
        for attr in ("local_head_witness", "local_tail_witness")
    ]
    return ChainedOptimizer([_make_distributed_optimizer(w, start=0, end=w.numel())[0] for w in weights])


class TestWitnessDumpAndClearStale:
    # [CLASS DOC] Tests of the dump-and-clear function over model chunks, with the event logger
    # getter and the parallel state getter patched (pipeline rank 0).
    def test_witness_dump_and_clear_stale_logs_all_witnesses(self) -> None:
        # [FUNCTION DOC] Two chunks with different nonzero rows. Asserts four log calls with
        # instance ids pp0_chunk0.local_head, pp0_chunk0.local_tail, pp0_chunk1.local_head,
        # pp0_chunk1.local_tail in order, each carrying the stale ids 5 and 6.
        """2 chunks x 2 witnesses = 4 log events with correct instance_ids."""
        chunk0 = _make_fake_chunk()
        chunk1 = _make_fake_chunk()
        chunk0.module.local_head_witness.witness.weight.data[1] = 1.0
        chunk0.module.local_tail_witness.witness.weight.data[2] = 1.0
        chunk1.module.local_head_witness.witness.weight.data[3] = 1.0
        chunk1.module.local_tail_witness.witness.weight.data[4] = 1.0

        model = [chunk0, chunk1]
        optimizer = _make_chained_optimizer_for_witnesses(model)
        witness_info = WitnessInfo(witness_ids=[1, 2, 3, 4], stale_ids=[5, 6])

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger, patch(
            "miles.utils.audit_utils.witness.module.get_parallel_state"
        ) as mock_get_parallel_state:
            mock_get_parallel_state.return_value.pp.rank = 0
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

            assert mock_logger.log.call_count == 4
            logged_instance_ids = [call[0][1]["instance_id"] for call in mock_logger.log.call_args_list]
            assert logged_instance_ids == [
                "pp0_chunk0.local_head",
                "pp0_chunk0.local_tail",
                "pp0_chunk1.local_head",
                "pp0_chunk1.local_tail",
            ]

            logged_stale_ids = [call[0][1]["stale_ids"] for call in mock_logger.log.call_args_list]
            for stale in logged_stale_ids:
                assert stale == [5, 6]

    def test_witness_dump_and_clear_stale_clears_stale_rows(self) -> None:
        # [FUNCTION DOC] All weights 1 and stale ids 3 and 7: afterwards both witnesses have rows
        # 3 and 7 equal to 0 and row 0 still 1.
        """Stale IDs should have their weight rows zeroed after the call."""
        chunk = _make_fake_chunk(buffer_size=10)
        chunk.module.local_head_witness.witness.weight.data.fill_(1.0)
        chunk.module.local_tail_witness.witness.weight.data.fill_(1.0)

        model = [chunk]
        optimizer = _make_chained_optimizer_for_witnesses(model)
        witness_info = WitnessInfo(witness_ids=[0], stale_ids=[3, 7])

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger, patch(
            "miles.utils.audit_utils.witness.module.get_parallel_state"
        ) as mock_get_parallel_state:
            mock_get_parallel_state.return_value.pp.rank = 0
            mock_get_logger.return_value = MagicMock()
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        for witness_attr in ("local_head_witness", "local_tail_witness"):
            witness = getattr(chunk.module, witness_attr)
            assert witness.witness.weight.data[3].item() == 0.0
            assert witness.witness.weight.data[7].item() == 0.0
            assert witness.witness.weight.data[0].item() == 1.0

    def test_witness_dump_and_clear_stale_empty_stale_ids(self) -> None:
        # [FUNCTION DOC] With no stale ids, all weights remain 1 for both witnesses.
        """Empty stale_ids should not trigger any zeroing."""
        chunk = _make_fake_chunk(buffer_size=10)
        chunk.module.local_head_witness.witness.weight.data.fill_(1.0)
        chunk.module.local_tail_witness.witness.weight.data.fill_(1.0)

        model = [chunk]
        optimizer = _make_chained_optimizer_for_witnesses(model)
        witness_info = WitnessInfo(witness_ids=[0], stale_ids=[])

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger, patch(
            "miles.utils.audit_utils.witness.module.get_parallel_state"
        ) as mock_get_parallel_state:
            mock_get_parallel_state.return_value.pp.rank = 0
            mock_get_logger.return_value = MagicMock()
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        for witness_attr in ("local_head_witness", "local_tail_witness"):
            witness = getattr(chunk.module, witness_attr)
            assert torch.all(witness.witness.weight.data == 1.0)

    def test_record_and_log_witness_param_includes_stale_ids(self) -> None:
        # [FUNCTION DOC] Calls the single-witness record function with stale ids 8 and 9 and
        # asserts the logged payload's stale ids equal [8, 9]. (Lives in this class though it
        # tests the record function directly.)
        """Log event should contain the correct stale_ids field."""
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[2] = 1.0

        with patch("miles.utils.audit_utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.head", stale_ids=[8, 9])

            mock_logger.log.assert_called_once()
            partial = mock_logger.log.call_args[0][1]
            assert partial["stale_ids"] == [8, 9]


class TestAbsBroadcastAddForward:
    # [CLASS DOC] Tests of the forward pass and shape checks of the absolute broadcast add
    # (hidden states plus an addend whose last dimension is 1).
    def test_forward_value_matches_plain_addition(self) -> None:
        # [FUNCTION DOC] Result equals hidden plus addend exactly.
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 2, 1)
        result = _abs_broadcast_add(hidden, addend)
        expected = hidden + addend
        assert torch.equal(result, expected)

    def test_forward_preserves_zero_addend(self) -> None:
        # [FUNCTION DOC] A zero addend leaves hidden states exactly equal.
        hidden = torch.randn(4, 2, 8)
        addend = torch.zeros(4, 2, 1)
        result = _abs_broadcast_add(hidden, addend)
        assert torch.equal(result, hidden)

    def test_forward_assert_addend_last_dim_must_be_1(self) -> None:
        # [FUNCTION DOC] An addend of last dim 3 raises an assertion error saying the last dim
        # must be 1.
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 2, 3)
        with pytest.raises(AssertionError, match="addend last dim must be 1"):
            _abs_broadcast_add(hidden, addend)

    def test_forward_assert_leading_dims_must_match(self) -> None:
        # [FUNCTION DOC] An addend whose second dim differs raises an assertion error saying all
        # dims except the last must match.
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 3, 1)  # second dim differs
        with pytest.raises(AssertionError, match="must match on all dims except last"):
            _abs_broadcast_add(hidden, addend)

    def test_forward_assert_ndim_must_match(self) -> None:
        # [FUNCTION DOC] A 2-dimensional addend with 3-dimensional hidden states raises the same
        # all-dims-must-match assertion error.
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(2, 1)  # 2D vs 3D
        with pytest.raises(AssertionError, match="must match on all dims except last"):
            _abs_broadcast_add(hidden, addend)


class TestAbsBroadcastAddBackwardHiddenStates:
    # [CLASS DOC] Tests of the hidden states gradient.
    def test_hidden_states_gradient_is_pass_through(self) -> None:
        # [FUNCTION DOC] Backward of the sum gives hidden states a gradient of all ones.
        hidden = torch.randn(4, 2, 8, requires_grad=True)
        addend = torch.randn(4, 2, 1, requires_grad=True)
        result = _abs_broadcast_add(hidden, addend)
        loss = result.sum()
        loss.backward()
        # hidden gradient = all ones (pass-through from sum)
        assert torch.equal(hidden.grad, torch.ones_like(hidden))


class TestAbsBroadcastAddBackwardAddend:
    # [CLASS DOC] Tests of the addend gradient, which is the sum of absolute upstream gradients
    # over the last dim.
    def test_addend_gradient_is_abs_sum_over_last_dim(self) -> None:
        # [FUNCTION DOC] With a fixed mixed-sign upstream gradient, the addend gradient is close to
        # the absolute value sum over the last dim (kept as a dim).
        hidden = torch.randn(3, 2, 4, requires_grad=True)
        addend = torch.zeros(3, 2, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        # Use a loss that produces a known gradient at result
        upstream_grad = torch.tensor(
            [
                [[1.0, -2.0, 3.0, -4.0], [0.5, -0.5, 0.5, -0.5]],
                [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                [[0.0, 0.0, 0.0, 0.0], [2.0, -1.0, 0.5, -0.3]],
            ]
        )
        result.backward(upstream_grad)

        # Expected addend grad: abs(upstream_grad).sum(dim=-1, keepdim=True)
        expected = upstream_grad.abs().sum(dim=-1, keepdim=True)
        assert torch.allclose(addend.grad, expected)

    def test_addend_gradient_no_cancellation_with_mixed_signs(self) -> None:
        # [FUNCTION DOC] An upstream gradient of alternating plus and minus ones sums to zero but
        # the addend gradient is exactly 8.
        """The key property: mixed-sign gradients don't cancel to zero."""
        hidden = torch.randn(1, 1, 8, requires_grad=True)
        addend = torch.zeros(1, 1, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        # Gradient with equal positive and negative values (sums to zero normally)
        upstream_grad = torch.tensor([[[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]])
        result.backward(upstream_grad)

        # Plain broadcast backward would give sum = 0
        assert upstream_grad.sum(dim=-1, keepdim=True).item() == 0.0
        # But abs broadcast gives sum of absolute values = 8
        assert addend.grad.item() == 8.0

    def test_addend_gradient_matches_plain_sum_when_all_positive(self) -> None:
        # [FUNCTION DOC] With an all-positive random upstream gradient, the addend gradient is
        # close to the plain sum over the last dim.
        """When all gradients are positive, abs().sum() == sum()."""
        hidden = torch.randn(2, 1, 4, requires_grad=True)
        addend = torch.zeros(2, 1, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        upstream_grad = torch.abs(torch.randn(2, 1, 4))  # all positive
        result.backward(upstream_grad)

        expected = upstream_grad.sum(dim=-1, keepdim=True)
        assert torch.allclose(addend.grad, expected)

    def test_addend_gradient_always_non_negative(self) -> None:
        # [FUNCTION DOC] With a random upstream gradient, every addend gradient entry is at least
        # zero.
        """abs().sum() is always >= 0."""
        hidden = torch.randn(10, 5, 16, requires_grad=True)
        addend = torch.zeros(10, 5, 1, requires_grad=True)

        result = _abs_broadcast_add(hidden, addend)
        upstream_grad = torch.randn(10, 5, 16)
        result.backward(upstream_grad)

        assert (addend.grad >= 0).all()


class TestAbsBroadcastAddGradientFlow:
    # [CLASS DOC] End-to-end gradient flow tests using a width-1 embedding as the witness source
    # (output is the embedding minus its detached copy, i.e. zero in value).
    def test_gradient_flows_through_to_embedding(self) -> None:
        # [FUNCTION DOC] Embeds ids 0 to 3, adds via the absolute broadcast add to random hidden
        # states, applies a linear output layer and a sum loss; asserts embedding rows 0 to 3 all
        # have nonzero gradient.
        """End-to-end: gradient reaches embedding weight via abs broadcast add."""
        vocab_size = 16
        hidden_dim = 8
        seq_len = 4

        embedding = torch.nn.Embedding(vocab_size, 1)
        torch.nn.init.zeros_(embedding.weight)
        output_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        witness_ids = torch.tensor([[0, 1, 2, 3]])  # [1, seq_len]
        w = embedding(witness_ids)  # [1, 4, 1]
        out = w - w.detach()

        hidden_states = torch.randn(seq_len, 1, hidden_dim, requires_grad=True)
        tail_out = out.transpose(0, 1).contiguous()  # [4, 1, 1]

        combined = _abs_broadcast_add(hidden_states, tail_out)
        logits = output_layer(combined)
        loss = logits.sum()
        loss.backward()

        # All 4 witness_ids should have nonzero gradient
        nonzero_rows = (embedding.weight.grad.abs() > 0).squeeze(-1)
        assert nonzero_rows[:4].all(), f"Expected rows 0-3 nonzero, got {embedding.weight.grad[:4]}"

    def test_gradient_nonzero_even_when_plain_broadcast_cancels(self) -> None:
        # [FUNCTION DOC] Uses an output layer weight whose rows each sum to 4 together with a
        # softmax cross entropy loss, a setup where a plain broadcast gradient cancels to zero.
        # Asserts the embedding gradient for witness rows 0 and 1 is nonzero with the absolute
        # broadcast add.
        """Simulates the exact scenario: output_layer gradient cancels under plain broadcast."""
        vocab_size = 4
        hidden_dim = 4
        seq_len = 2

        embedding = torch.nn.Embedding(8, 1)
        torch.nn.init.zeros_(embedding.weight)

        # Construct output_layer weight where ROW sums are constant (each row sums to 4),
        # so that dL/d_combined.sum_over_last_dim = sum_i dL/dlogit_i * row_sum_i = 0
        # (using softmax-loss sum-to-zero property). Row values differ so per-element
        # grads are nonzero — only their sum cancels. This is exactly the scenario
        # where plain broadcast (sum) cancels but abs broadcast (abs.sum) preserves.
        W = torch.tensor([[2.0, 0.0, 1.0, 1.0], [0.0, 2.0, 1.0, 1.0], [1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 0.0, 2.0]])
        output_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        output_layer.weight.data = W

        witness_ids = torch.tensor([[0, 1]])
        w = embedding(witness_ids)
        out = w - w.detach()

        hidden_states = torch.randn(seq_len, 1, hidden_dim, requires_grad=True)
        tail_out = out.transpose(0, 1).contiguous()

        # With abs broadcast add, gradient should be nonzero
        combined = _abs_broadcast_add(hidden_states, tail_out)
        logits = output_layer(combined)
        targets = torch.tensor([[0, 1]])
        log_probs = torch.nn.functional.log_softmax(logits.squeeze(1), dim=-1)
        loss = -log_probs.gather(1, targets.T).sum()
        loss.backward()

        assert embedding.weight.grad is not None
        assert (
            embedding.weight.grad[:2].abs() > 0
        ).all(), f"Expected nonzero grad for witness rows 0,1, got {embedding.weight.grad[:2]}"


class TestAbsBroadcastAddDoubleBackward:
    # [CLASS DOC] Numerical gradient verification of the autograd function. (Class name mentions
    # double backward but only a first-order gradient check is run.)
    def test_gradcheck(self) -> None:
        # [FUNCTION DOC] Runs the numerical gradient check in double precision for the hidden
        # states argument only (the addend is detached, since the absolute value is not
        # differentiable at zero).
        """Verify numerical gradient correctness with torch.autograd.gradcheck."""
        hidden = torch.randn(2, 2, 4, dtype=torch.float64, requires_grad=True)
        addend = torch.randn(2, 2, 1, dtype=torch.float64, requires_grad=True)
        # gradcheck only tests hidden_states gradient (which is pass-through)
        # For addend, abs() is not differentiable at 0, so we test separately
        assert torch.autograd.gradcheck(
            lambda h: _AbsBroadcastAdd.apply(h, addend.detach()),
            (hidden,),
        )
