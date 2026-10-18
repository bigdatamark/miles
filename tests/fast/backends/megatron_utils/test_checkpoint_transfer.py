# [MODULE DOC] CPU tests of the Megatron backend's fault-tolerance checkpoint transfer: the
# transport codec that wraps a tensor-aware state dict plus iteration into a payload (separating
# tensors into deduplicated flat byte storages with per-tensor view metadata, and a hollow state-dict
# shell) and unwraps it again, including a simulation of the process-group transport's pytree
# flatten / pickle / unflatten wire protocol; the structured start/end log records emitted by the
# send and receive entry points (using a fake transport); and an extensive suite for the tensor-view
# codec (storage dedup, storage id order, dtypes, strides, offsets, scalars, empty tensors,
# parameters, aliasing without copies, clone-before-decode, and CUDA / mixed-device cases).
import logging
import pickle
from types import SimpleNamespace

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
from torch.utils._pytree import tree_flatten_with_path, tree_unflatten

from miles.backends.megatron_utils.ft import checkpoint_transfer, in_memory_checkpoint
from miles.backends.megatron_utils.ft.checkpoint_transfer import _TensorViewCodec, _TransportCodec
from miles.utils.ft_utils.process_group_utils import GroupInfo

_CKPT_TRANSFER_LOGGER = "miles.backends.megatron_utils.ft.checkpoint_transfer"


class _FakeTransport:
    # [CLASS DOC] HELPER: Fake checkpoint transport recording what is sent (payloads and destination
    # rank lists, and a count of disallow calls) and returning a preset payload on receive.
    def __init__(self, recv_payload: dict | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the payload to return on receive and initializes empty
        # records of sent payloads and destinations and a zero disallow counter.
        self.recv_payload = recv_payload
        self.sent_payloads: list[dict] = []
        self.sent_dst_ranks: list[list[int]] = []
        self.disallow_calls = 0

    def send_checkpoint(self, *, dst_ranks: list[int], step: int, state_dict: dict, timeout) -> None:
        # [FUNCTION DOC] HELPER: Records the destination ranks and the state dict (payload); step and
        # timeout are ignored.
        self.sent_dst_ranks.append(dst_ranks)
        self.sent_payloads.append(state_dict)

    def disallow_checkpoint(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the disallow-call counter.
        self.disallow_calls += 1

    def metadata(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the constant string "fake-metadata".
        return "fake-metadata"

    def recv_checkpoint(self, *, src_rank: int, metadata: str, step: int, timeout) -> dict:
        # [FUNCTION DOC] HELPER: Ignores all arguments and returns the preset receive payload.
        return self.recv_payload


def _transfer_messages(caplog) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the formatted messages, in order, of captured log records whose
    # logger name is the checkpoint-transfer module's logger.
    return [record.getMessage() for record in caplog.records if record.name == _CKPT_TRANSFER_LOGGER]


@pytest.fixture()
def state_dict() -> MCoreTensorAwareStateDict:
    # [FUNCTION DOC] HELPER: Fixture returning a small tensor-aware state dict with three sharded
    # tensors: model layer1 weight (float32, 4 by 8 arange), model layer2 weight (float32, 2 by 6 of
    # 7.0), and optimizer step (int64, one element 100), with common data holding iteration 0 and a
    # placeholder args representation.
    sharded_state_dict: dict = {
        "model": {
            "layer1.weight": ShardedTensor.from_rank_offsets(
                "layer1.weight", torch.arange(32, dtype=torch.float32).reshape(4, 8)
            ),
            "layer2.weight": ShardedTensor.from_rank_offsets("layer2.weight", torch.full((2, 6), fill_value=7.0)),
        },
        "optimizer": {
            "step": ShardedTensor.from_rank_offsets("step", torch.tensor([100], dtype=torch.int64)),
        },
    }
    common = {"iteration": 0, "args_repr": "dummy"}
    return MCoreTensorAwareStateDict(common=common, sharded_state_dict=sharded_state_dict)


class TestSerializeForTransport:
    # [CLASS DOC] Tests of the transport codec's encode step.
    def test_returns_separated_storages_iteration_and_hollow_shell(self, state_dict: MCoreTensorAwareStateDict):
        # [FUNCTION DOC] Encodes the fixture with iteration 42 and asserts: the payload's iteration is
        # 42; unique storages and view metadata are lists with 3 view entries; decoding them with the
        # view codec reproduces the original tensors exactly; and the payload's hollow state dict is the
        # same object as the input and is marked hollow (its tensors were popped).
        original_tensors = [t.clone() for t in state_dict.tensors]

        payload = _TransportCodec.encode(state_dict=state_dict, iteration=42)

        assert payload["iteration"] == 42
        assert isinstance(payload["unique_storages"], list)
        assert isinstance(payload["view_metas"], list)
        assert len(payload["view_metas"]) == 3

        # Round-trip the storages+metas back into views and compare.
        decoded = _TensorViewCodec.decode(payload["unique_storages"], payload["view_metas"])
        for t, original in zip(decoded, original_tensors, strict=True):
            assert torch.equal(t, original)

        assert payload["hollow_state_dict"] is state_dict
        assert payload["hollow_state_dict"].is_hollow

    def test_pytree_flatten_yields_each_tensor_as_separate_leaf(self, state_dict: MCoreTensorAwareStateDict):
        # [FUNCTION DOC] Flattens the encoded payload with the pytree utility and asserts there are 3
        # tensor leaves, the non-tensor leaves include the hollow state dict object, and include the
        # integer 42.
        """The whole point of the fix: PGTransport's tree_flatten_with_path must see
        each ShardedTensor.data as its own leaf, not buried inside a pickled blob."""
        payload = _TransportCodec.encode(state_dict=state_dict, iteration=42)

        leaves, _ = tree_flatten_with_path(payload)
        tensor_leaves = [v for _, v in leaves if isinstance(v, torch.Tensor)]
        non_tensor_leaves = [v for _, v in leaves if not isinstance(v, torch.Tensor)]

        assert len(tensor_leaves) == 3
        assert any(isinstance(v, MCoreTensorAwareStateDict) for v in non_tensor_leaves)
        assert 42 in non_tensor_leaves

    def test_hollow_shell_pickles_without_dragging_tensor_data(self, state_dict: MCoreTensorAwareStateDict):
        # [FUNCTION DOC] Pickles and unpickles the encoded payload's hollow state dict; asserts the
        # restored object is hollow, still lists 3 sharded tensors, each with no data but still with a
        # recorded original device attribute.
        """PGTransport pickles non-tensor leaves; the hollow shell must survive
        a pickle round-trip and not contain any of the original tensor storage."""
        payload = _TransportCodec.encode(state_dict=state_dict, iteration=42)

        restored = pickle.loads(pickle.dumps(payload["hollow_state_dict"]))

        assert restored.is_hollow
        sharded_tensors = list(restored._sharded_tensors)
        assert len(sharded_tensors) == 3
        assert all(sh.data is None for sh in sharded_tensors)
        assert all(hasattr(sh, "orig_device") for sh in sharded_tensors)


class TestDeserializeFromTransport:
    # [CLASS DOC] Tests of the transport codec's decode step.
    def test_round_trip_preserves_tensor_values_iteration_and_common(self, state_dict: MCoreTensorAwareStateDict):
        # [FUNCTION DOC] Encodes then decodes in-process; asserts the iteration is 42, the state dict
        # is no longer hollow, the common data equals the original, and tensors equal the originals
        # exactly in order.
        original_tensors = [t.clone() for t in state_dict.tensors]
        original_common = dict(state_dict.common)

        payload = _TransportCodec.encode(state_dict=state_dict, iteration=42)
        iteration_back, state_dict_back = _TransportCodec.decode(payload)

        assert iteration_back == 42
        assert not state_dict_back.is_hollow
        assert state_dict_back.common == original_common
        for original, back in zip(original_tensors, state_dict_back.tensors, strict=True):
            assert torch.equal(original, back)

    def test_full_pgtransport_simulation_round_trip(self, state_dict: MCoreTensorAwareStateDict):
        # [FUNCTION DOC] Simulates the wire protocol in five steps: encode; flatten the payload with
        # the pytree utility; pickle the tree spec and non-tensor leaves while cloning tensor leaves (as
        # a network receive would); on the receive side unpickle and interleave tensors and non-tensor
        # values back into leaf order using a tensor mask; unflatten and decode. Asserts the same
        # properties as the in-process round trip (iteration, not hollow, common data, tensor values).
        """End-to-end simulation of PGTransport: pytree flatten on sender, pickle
        non-tensor leaves + treespec, clone tensor leaves to mimic NCCL transfer,
        unflatten on receiver. Verifies our (de)serializers survive the actual
        wire protocol — not just an in-process pop/insert."""
        original_tensors = [t.clone() for t in state_dict.tensors]
        original_common = dict(state_dict.common)

        # Step 1: sender — wrap state_dict for transport
        payload = _TransportCodec.encode(state_dict=state_dict, iteration=42)

        # Step 2: sender — flatten via pytree (what PGTransport does internally)
        leaves, treespec = tree_flatten_with_path(payload)

        # Step 3: sender — pickle treespec + non-tensor leaves; "send" tensor leaves over the wire
        is_tensor_mask: list[bool] = [isinstance(v, torch.Tensor) for _, v in leaves]
        pickled_metadata = pickle.dumps(
            (treespec, [v for v, m in zip([v for _, v in leaves], is_tensor_mask, strict=True) if not m])
        )
        wire_tensors = [v.clone() for v, m in zip([v for _, v in leaves], is_tensor_mask, strict=True) if m]

        # Step 4: receiver — unpickle metadata + interleave received tensors back into leaf order
        treespec_recv, non_tensor_values = pickle.loads(pickled_metadata)
        recv_values: list = []
        ti = 0
        nti = 0
        for is_tensor in is_tensor_mask:
            if is_tensor:
                recv_values.append(wire_tensors[ti])
                ti += 1
            else:
                recv_values.append(non_tensor_values[nti])
                nti += 1
        payload_recv = tree_unflatten(recv_values, treespec_recv)

        # Step 5: receiver — unwrap into iteration + state_dict
        iteration_back, state_dict_back = _TransportCodec.decode(payload_recv)

        assert iteration_back == 42
        assert not state_dict_back.is_hollow
        assert state_dict_back.common == original_common
        for original, back in zip(original_tensors, state_dict_back.tensors, strict=True):
            assert torch.equal(original, back)


class TestSendCkptRecords:
    # [CLASS DOC] Tests of the log records emitted by the send entry point.
    def test_send_brackets_the_transfer_with_ft_tagged_records(self, state_dict, monkeypatch, caplog):
        # [FUNCTION DOC] Replaces the module's save-to-memory function with one returning the fixture
        # and its transport factory with one returning a fake transport. Calls send with a two-member
        # independent-DP group info, iteration 7 and destination rank 3 under info-level log capture.
        # Asserts the fake transport got destination list [3] and a payload with iteration 7, and that
        # exactly two records were logged by the module, the start then end cross-cell checkpoint-send
        # messages naming iteration 7 and destination alive rank 3.
        """send_ckpt brackets the transport send with ft-tagged ckpt_send records carrying iteration and dst rank."""
        transport = _FakeTransport()
        monkeypatch.setattr(checkpoint_transfer, "save_to_memory", lambda **kwargs: state_dict)
        monkeypatch.setattr(checkpoint_transfer, "_create_transport", lambda indep_dp, timeout: transport)

        with caplog.at_level(logging.INFO, logger=_CKPT_TRANSFER_LOGGER):
            checkpoint_transfer.send_ckpt(
                indep_dp=GroupInfo(rank=0, size=2, group=None),
                model=[],
                optimizer=object(),
                opt_param_scheduler=object(),
                iteration=7,
                dst_rank=3,
            )

        assert transport.sent_dst_ranks == [[3]]
        assert transport.sent_payloads[0]["iteration"] == 7
        assert _transfer_messages(caplog) == [
            "ft op=cross_cell phase=start kind=ckpt_send iteration=7 to_alive_rank=3",
            "ft op=cross_cell phase=end kind=ckpt_send iteration=7 to_alive_rank=3",
        ]


class TestRecvCkptRecords:
    # [CLASS DOC] Tests of the log records emitted by the receive entry point.
    def test_recv_brackets_the_transfer_with_ft_tagged_records(self, state_dict, monkeypatch, caplog):
        # [FUNCTION DOC] Builds a fake transport that returns a payload encoded from the fixture with
        # iteration 11, installs it as the module's transport, and patches the in-memory checkpoint
        # module's argument getter to return a local non-persistent checkpoint type with the fully
        # parallel algorithm. Receives from source rank 2 on rank 1 of 2 under info-level capture;
        # asserts the returned manager's latest iteration is 11 and exactly two records were logged: the
        # start message (source alive rank 2) and the end message (iteration 11, source alive rank 2).
        """recv_ckpt brackets the transport recv with ft-tagged ckpt_recv records naming the source rank."""
        transport = _FakeTransport(recv_payload=_TransportCodec.encode(state_dict=state_dict, iteration=11))
        monkeypatch.setattr(checkpoint_transfer, "_create_transport", lambda indep_dp, timeout: transport)
        monkeypatch.setattr(
            in_memory_checkpoint,
            "get_args",
            lambda: SimpleNamespace(
                non_persistent_ckpt_type="local",
                non_persistent_local_ckpt_algo="fully_parallel",
            ),
        )

        with caplog.at_level(logging.INFO, logger=_CKPT_TRANSFER_LOGGER):
            manager = checkpoint_transfer.recv_ckpt(indep_dp=GroupInfo(rank=1, size=2, group=None), src_rank=2)

        assert manager.find_latest() == 11
        assert _transfer_messages(caplog) == [
            "ft op=cross_cell phase=start kind=ckpt_recv from_alive_rank=2",
            "ft op=cross_cell phase=end kind=ckpt_recv iteration=11 from_alive_rank=2",
        ]


class TestTensorViewCodec:
    # [CLASS DOC] Tests of the tensor-view codec that encodes a list of tensors into a list of
    # deduplicated flat byte storages plus per-tensor metadata (storage id, dtype, shape, stride,
    # storage offset) and decodes them back into views aliasing the storages. Storage ids are
    # assigned in first-seen order, keyed by device and storage address.
    """Comprehensive UT for `_TensorViewCodec.encode/decode`.

    Round-trip semantics: `decode(*encode(tensors))` must produce a list of
    tensors that are *value-equal* to the inputs, while sharing storage with
    the encoded `unique_storages` (the whole point of the codec).
    """

    def test_empty_input_yields_empty_output(self):
        # [FUNCTION DOC] Encoding an empty list gives empty storages and metadata, and decoding those
        # gives an empty list.
        unique_storages, view_metas = _TensorViewCodec.encode([])
        assert unique_storages == []
        assert view_metas == []
        assert _TensorViewCodec.decode(unique_storages, view_metas) == []

    def test_single_tensor_round_trip(self):
        # [FUNCTION DOC] A 3 by 4 float32 arange encodes to one storage and one metadata entry with
        # storage id 0, float32 dtype, shape (3, 4), stride (4, 1), offset 0; decoding gives an equal
        # tensor with the same dtype and shape.
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        unique_storages, view_metas = _TensorViewCodec.encode([original])

        assert len(unique_storages) == 1
        assert len(view_metas) == 1
        assert view_metas[0]["storage_id"] == 0
        assert view_metas[0]["dtype"] == torch.float32
        assert view_metas[0]["shape"] == (3, 4)
        assert view_metas[0]["stride"] == (4, 1)
        assert view_metas[0]["storage_offset"] == 0

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert len(decoded) == 1
        assert torch.equal(decoded[0], original)
        assert decoded[0].dtype == original.dtype
        assert decoded[0].shape == original.shape

    def test_distinct_storages_yield_distinct_storage_ids(self):
        # [FUNCTION DOC] Three independent tensors (float32, int64, float64) give three storages with
        # ids 0, 1, 2 in order, and decode back to equal tensors.
        a = torch.arange(8, dtype=torch.float32)
        b = torch.arange(4, dtype=torch.int64)
        c = torch.zeros(5, dtype=torch.float64)

        unique_storages, view_metas = _TensorViewCodec.encode([a, b, c])

        assert len(unique_storages) == 3
        assert [vm["storage_id"] for vm in view_metas] == [0, 1, 2]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], a)
        assert torch.equal(decoded[1], b)
        assert torch.equal(decoded[2], c)

    def test_shared_storage_dedups_into_one_unique_storage(self):
        # [FUNCTION DOC] Three views of a 100-element base (slice at offset 10 length 10, a 4 by 10
        # view from offset 20, and a slice at offset 0 length 5) produce a single storage, all storage ids
        # 0, metadata recording the distinct offsets and shapes, and exact value equality after decode.
        """The dedup invariant: N views over one storage produce 1 unique_storage."""
        base = torch.arange(100, dtype=torch.float32)
        view_a = base[10:20]  # offset=10, length=10
        view_b = base[20:60].view(4, 10)  # offset=20, shape=(4, 10)
        view_c = base[:5]  # offset=0, length=5

        unique_storages, view_metas = _TensorViewCodec.encode([view_a, view_b, view_c])

        assert len(unique_storages) == 1
        assert all(vm["storage_id"] == 0 for vm in view_metas)
        # storage_offset / shape / stride encode the difference
        assert view_metas[0]["storage_offset"] == 10
        assert view_metas[0]["shape"] == (10,)
        assert view_metas[1]["storage_offset"] == 20
        assert view_metas[1]["shape"] == (4, 10)
        assert view_metas[2]["storage_offset"] == 0
        assert view_metas[2]["shape"] == (5,)

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], view_a)
        assert torch.equal(decoded[1], view_b)
        assert torch.equal(decoded[2], view_c)

    def test_partial_dedup_preserves_storage_id_assignment_order(self):
        # [FUNCTION DOC] Encodes five tensors where the third is a view of the first and the fifth a
        # view of the second; asserts three storages and storage ids [0, 1, 0, 2, 1].
        """First-seen wins: storage_id increments only on a NEW storage_ptr."""
        s1 = torch.arange(10, dtype=torch.float32)  # storage A
        s2 = torch.arange(20, dtype=torch.int32)  # storage B
        s1_view = s1[2:8]  # storage A again
        s3 = torch.zeros(4, dtype=torch.float64)  # storage C
        s2_view = s2[5:10]  # storage B again

        unique_storages, view_metas = _TensorViewCodec.encode([s1, s2, s1_view, s3, s2_view])

        assert len(unique_storages) == 3
        assert [vm["storage_id"] for vm in view_metas] == [0, 1, 0, 2, 1]

    def test_dtype_preserved_across_round_trip(self):
        # [FUNCTION DOC] For ten dtypes (float 16/32/64, bfloat16, int 8/16/32/64, uint8, bool), a
        # 5-element zeros tensor of each round-trips with the same dtype and equal values.
        dtypes = [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
            torch.bfloat16,
        ]
        tensors = [torch.zeros(5, dtype=dt) for dt in dtypes]

        unique_storages, view_metas = _TensorViewCodec.encode(tensors)
        decoded = _TensorViewCodec.decode(unique_storages, view_metas)

        for d, original in zip(decoded, tensors, strict=True):
            assert d.dtype == original.dtype
            assert torch.equal(d, original)

    def test_non_contiguous_tensor_preserves_stride_and_values(self):
        # [FUNCTION DOC] A transposed 3 by 4 view (asserted non-contiguous) encodes with its shape and
        # its transposed stride in the metadata, and decodes to a tensor with the same stride and equal
        # values.
        """Transposed view: shape=(3,4), stride=(1,3) (column-major over storage)."""
        base = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        transposed = base.t()  # shape=(3,4), non-contiguous
        assert not transposed.is_contiguous()

        unique_storages, view_metas = _TensorViewCodec.encode([transposed])

        assert view_metas[0]["shape"] == (3, 4)
        assert view_metas[0]["stride"] == transposed.stride()

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].stride() == transposed.stride()
        assert torch.equal(decoded[0], transposed)

    def test_storage_offset_preserved_for_slice_view(self):
        # [FUNCTION DOC] A slice at index 7 of a 20-element tensor has storage offset 7 in the
        # metadata, and the decoded view has the same offset and equal values.
        base = torch.arange(20, dtype=torch.float32)
        sliced = base[7:15]
        assert sliced.storage_offset() == 7

        unique_storages, view_metas = _TensorViewCodec.encode([sliced])

        assert view_metas[0]["storage_offset"] == 7

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].storage_offset() == 7
        assert torch.equal(decoded[0], sliced)

    def test_decoded_view_shares_storage_with_unique_storage(self):
        # [FUNCTION DOC] Zeroing the encoded byte storage in place makes the decoded float tensor
        # all zeros, proving decode returns aliasing views, not copies.
        """Decoded views must alias the unique_storage (no copy)."""
        original = torch.arange(8, dtype=torch.float32)

        unique_storages, view_metas = _TensorViewCodec.encode([original])
        decoded = _TensorViewCodec.decode(unique_storages, view_metas)

        # Mutating the unique_storage (uint8 view) must propagate to decoded view.
        unique_storages[0].zero_()
        assert torch.equal(decoded[0], torch.zeros(8, dtype=torch.float32))

    def test_encode_aliases_input_storage_no_copy(self):
        # [FUNCTION DOC] Zeroing the original tensor in place makes the sum of the encoded byte storage
        # zero, proving encode does not copy.
        """Encoded uint8 storage must alias the input tensor's storage."""
        original = torch.arange(8, dtype=torch.float32)

        unique_storages, _ = _TensorViewCodec.encode([original])

        # Mutating the input tensor must propagate to the encoded uint8 view.
        original.zero_()
        assert unique_storages[0].sum().item() == 0

    def test_multi_dtype_views_into_same_storage(self):
        # [FUNCTION DOC] A float32 tensor and its int32 reinterpretation (same storage) encode to one
        # storage with metadata dtypes float32 and int32, and decode to tensors equal to each input.
        """A storage can be viewed at multiple dtypes (e.g. fp32 vs int32)."""
        base = torch.arange(8, dtype=torch.float32)
        as_int = base.view(torch.int32)  # same storage, different dtype
        assert base.untyped_storage().data_ptr() == as_int.untyped_storage().data_ptr()

        unique_storages, view_metas = _TensorViewCodec.encode([base, as_int])

        assert len(unique_storages) == 1
        assert view_metas[0]["dtype"] == torch.float32
        assert view_metas[1]["dtype"] == torch.int32

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], as_int)

    def test_round_trip_idempotent_under_repeated_encoding(self):
        # [FUNCTION DOC] Encodes, decodes, and encodes the decoded tensors again; asserts both runs
        # produce one storage and identical metadata.
        """encode(decode(encode(t))) == encode(t)."""
        original = torch.arange(20, dtype=torch.float32).reshape(4, 5)

        s1, m1 = _TensorViewCodec.encode([original])
        decoded = _TensorViewCodec.decode(s1, m1)
        s2, m2 = _TensorViewCodec.encode(decoded)

        assert len(s1) == len(s2) == 1
        assert m1 == m2

    def test_round_trip_property_random_mix(self):
        # [FUNCTION DOC] With seed 0, encodes six tensors: two random float tensors, two views of one
        # shared 64-element base, an int64 random tensor and an empty float tensor. Asserts there are six
        # metadata entries, five unique storages, and that decoded tensors match in dtype, shape and
        # values.
        """Property: for a random mix of tensors, decode round-trip is value-preserving."""
        torch.manual_seed(0)
        shared_base = torch.randn(64)
        tensors = [
            torch.randn(3, 4),
            torch.randn(7),
            shared_base[10:30],
            shared_base[5:15].view(2, 5),
            torch.randint(0, 100, (4, 4), dtype=torch.int64),
            torch.zeros(0, dtype=torch.float32),  # empty tensor
        ]

        unique_storages, view_metas = _TensorViewCodec.encode(tensors)

        assert len(view_metas) == len(tensors)
        # shared_base appears as 1 unique storage even though 2 views reference it
        assert len(unique_storages) == 5

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        for d, original in zip(decoded, tensors, strict=True):
            assert d.dtype == original.dtype
            assert d.shape == original.shape
            assert torch.equal(d, original)

    def test_decode_after_storage_clone_preserves_values(self):
        # [FUNCTION DOC] Encodes a 4 by 4 tensor and a slice of it, clones the storages (as a network
        # receive would), decodes from the clones, and asserts values match the inputs. Then zeroes the
        # cloned storage and asserts both decoded views are zero while the original input is not.
        """Simulates wire transfer: clone unique_storages (mimic NCCL recv copy)
        before decoding. Decoded views must still produce correct values, and
        be aliased to the clone (not the original)."""
        original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        sliced = original[1:3]

        unique_storages, view_metas = _TensorViewCodec.encode([original, sliced])
        cloned_storages = [u.clone() for u in unique_storages]

        decoded = _TensorViewCodec.decode(cloned_storages, view_metas)
        assert torch.equal(decoded[0], original)
        assert torch.equal(decoded[1], sliced)

        # Mutate the clone — decoded views follow the clone, not the original.
        cloned_storages[0].zero_()
        assert decoded[0].sum().item() == 0
        assert decoded[1].sum().item() == 0
        assert original.sum().item() != 0  # original untouched

    def test_same_tensor_twice_dedups_into_one_storage(self):
        # [FUNCTION DOC] Encoding the same tensor object twice gives one storage, storage ids [0, 0],
        # equal metadata entries, and two equal decoded tensors.
        """encode([t, t]) — two references to the same tensor share one storage."""
        t = torch.arange(8, dtype=torch.float32)

        unique_storages, view_metas = _TensorViewCodec.encode([t, t])

        assert len(unique_storages) == 1
        assert [vm["storage_id"] for vm in view_metas] == [0, 0]
        assert view_metas[0] == view_metas[1]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], t)
        assert torch.equal(decoded[1], t)

    def test_zero_dim_scalar_round_trip(self):
        # [FUNCTION DOC] A 0-d float tensor (7.5) has empty shape and stride in the metadata and
        # round-trips to a 0-d tensor that is equal and whose item is 7.5.
        """0-d (scalar) tensor: shape=(), stride=(), storage_offset=0."""
        scalar = torch.tensor(7.5, dtype=torch.float32)
        assert scalar.shape == ()
        assert scalar.stride() == ()

        unique_storages, view_metas = _TensorViewCodec.encode([scalar])

        assert view_metas[0]["shape"] == ()
        assert view_metas[0]["stride"] == ()

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].shape == ()
        assert torch.equal(decoded[0], scalar)
        assert decoded[0].item() == 7.5

    def test_nn_parameter_input(self):
        # [FUNCTION DOC] A 2 by 3 module parameter encodes to one storage with float32 dtype and shape
        # (2, 3), and decodes to a tensor equal to the parameter's data.
        """Production input via state_dict.pop_tensors() may include nn.Parameter.
        Codec uses .untyped_storage() which works on Parameter same as Tensor."""
        param = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))

        unique_storages, view_metas = _TensorViewCodec.encode([param])

        assert len(unique_storages) == 1
        assert view_metas[0]["dtype"] == torch.float32
        assert view_metas[0]["shape"] == (2, 3)

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], param.data)

    def test_empty_slice_of_nonempty_storage(self):
        # [FUNCTION DOC] An empty slice (index 5 to 5) of a 20-element base is first asserted to share
        # the base's storage address. Encoding base and empty view gives one storage, ids [0, 0], the
        # empty view's metadata with shape (0,) and offset 5; decoding preserves the base's shape and the
        # empty view's shape and values.
        """Zero-length view (shape=(0,)) over a non-empty storage."""
        base = torch.arange(20, dtype=torch.float32)
        empty_view = base[5:5]
        assert empty_view.shape == (0,)
        # Empty view shares its base's storage (data_ptr is non-null).
        assert empty_view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()

        unique_storages, view_metas = _TensorViewCodec.encode([base, empty_view])

        assert len(unique_storages) == 1
        assert [vm["storage_id"] for vm in view_metas] == [0, 0]
        assert view_metas[1]["shape"] == (0,)
        assert view_metas[1]["storage_offset"] == 5

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].shape == base.shape
        assert decoded[1].shape == (0,)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], empty_view)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_round_trip(self):
        # [FUNCTION DOC] Skipped without CUDA. Encodes a 64-element GPU tensor, two views of it and an
        # independent bfloat16 GPU tensor; asserts two storages on the GPU, ids [0, 0, 0, 1], decoded
        # tensors on the GPU and equal to the inputs.
        """Production runs on CUDA — verify the codec round-trips on GPU tensors,
        including shared-storage dedup and dtype preservation."""
        base = torch.arange(64, dtype=torch.float32, device="cuda")
        view_a = base[10:30]  # shares storage with base
        view_b = base[30:50].view(4, 5)  # shares storage with base
        independent = torch.zeros(8, dtype=torch.bfloat16, device="cuda")

        unique_storages, view_metas = _TensorViewCodec.encode([base, view_a, view_b, independent])

        assert len(unique_storages) == 2
        assert all(s.device.type == "cuda" for s in unique_storages)
        assert [vm["storage_id"] for vm in view_metas] == [0, 0, 0, 1]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert all(d.device.type == "cuda" for d in decoded)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], view_a)
        assert torch.equal(decoded[2], view_b)
        assert torch.equal(decoded[3], independent)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_storage_aliasing_no_copy(self):
        # [FUNCTION DOC] Skipped without CUDA. Encodes and decodes a GPU tensor, asserts the storage
        # is on the GPU, then zeroes the input in place and asserts the decoded view sums to zero (the
        # chain aliases device memory).
        """On CUDA, the encoded uint8 storage must alias the input tensor's
        device memory (no host-device or device-device copy)."""
        original = torch.arange(8, dtype=torch.float32, device="cuda")

        unique_storages, view_metas = _TensorViewCodec.encode([original])
        decoded = _TensorViewCodec.decode(unique_storages, view_metas)

        assert unique_storages[0].device.type == "cuda"
        # Mutate input — should propagate through the encoded storage to decoded view.
        original.zero_()
        assert decoded[0].sum().item() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_mixed_device_tensors_get_distinct_storages(self):
        # [FUNCTION DOC] Skipped without CUDA. Encodes a CPU and a GPU tensor with equal contents;
        # asserts two storages (ids 0 and 1) on the CPU and GPU respectively, and decoded tensors keep
        # their devices and equal their inputs.
        """CPU and CUDA tensors never share a storage entry, and each decoded view keeps its device."""
        cpu_t = torch.arange(8, dtype=torch.float32)
        cuda_t = torch.arange(8, dtype=torch.float32, device="cuda")

        unique_storages, view_metas = _TensorViewCodec.encode([cpu_t, cuda_t])

        assert len(unique_storages) == 2
        assert [vm["storage_id"] for vm in view_metas] == [0, 1]
        assert unique_storages[0].device.type == "cpu"
        assert unique_storages[1].device.type == "cuda"

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert decoded[0].device.type == "cpu"
        assert decoded[1].device.type == "cuda"
        assert torch.equal(decoded[0], cpu_t)
        assert torch.equal(decoded[1], cuda_t)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_mixed_device_dedup_stays_per_device(self):
        # [FUNCTION DOC] Skipped without CUDA. Encodes a GPU base, a view of it and a CPU tensor;
        # asserts two storages, ids [0, 0, 1], and that all three decode to values equal to their inputs.
        """Views sharing storage dedup within their device while the other device's tensor stays separate."""
        base = torch.arange(16, dtype=torch.float32, device="cuda")
        view = base[4:12]
        cpu_t = torch.arange(4, dtype=torch.float32)

        unique_storages, view_metas = _TensorViewCodec.encode([base, view, cpu_t])

        assert len(unique_storages) == 2
        assert [vm["storage_id"] for vm in view_metas] == [0, 0, 1]

        decoded = _TensorViewCodec.decode(unique_storages, view_metas)
        assert torch.equal(decoded[0], base)
        assert torch.equal(decoded[1], view)
        assert torch.equal(decoded[2], cpu_t)
