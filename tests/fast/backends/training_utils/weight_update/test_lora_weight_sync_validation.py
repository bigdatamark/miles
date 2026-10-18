# [MODULE DOC]
# CPU tests of LoRA weight sync validation: the engine result checker (raises on failure results,
# with different messages for LoRA and base syncs, accepts plain tuple results), the updater's
# handling of a base model whose weight iterator yields zero buckets (no send, post-base hook called
# once, weight version becomes one), and flattened tensor bucket round trips (single dtype, mixed
# dtypes expected to fail, receiver-side rebuild from flat tensor plus metadata, LoRA-only filter).
# The bucket tests skip when SGLang's bucket class is not importable.
# KNOWN ISSUE: the module docstring promises coverage of distributed NCCL adapter broadcast but no
# test here covers it, and the module-path constants for the CUDA IPC and broadcast protocols are
# defined but unused. The docstring also points at a test file path that is not where the iterator
# tests live (they are in the weight_update subdirectory).
"""Tests for LoRA weight-sync validation logic.

Verifies that silent failures are caught:
- Engine returning success=False raises RuntimeError
- Iterator validation errors (empty export / no lora names) propagate through
  the orchestration (the guards themselves are covered in
  tests/fast/backends/training_utils/test_hf_weight_iterator.py)
- FlattenedTensorBucket round-trip preserves tensor values
- Distributed (disaggregate) sync broadcasts the adapter over NCCL (no CUDA IPC)
"""

from argparse import Namespace
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.lora.utils import is_lora_weight_name

_UW_MODULE = "miles.backends.training_utils.weight_update.protocols.cuda_ipc"
_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"
_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_LORA_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
    ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
    ("model.layers.0.mlp.gate_proj.lora_A.weight", torch.randn(8, 2)),
    ("model.layers.0.mlp.gate_proj.lora_B.weight", torch.randn(2, 8)),
]


@dataclass
class _FakeEngineResult:
    # [CLASS DOC] HELPER: Simple record imitating an engine weight-sync result: a success flag and
    # an optional error message (default none).
    """Mimics sglang's LoRAUpdateOutput / weight-sync result."""

    success: bool
    error_message: str | None = None


def _make_args(**overrides):
    # [FUNCTION DOC] HELPER: Returns an argument namespace with defaults (LoRA rank 32, alpha 32,
    # dropout 0, two target modules, bridge conversion mode, one GPU per engine, a fake checkpoint
    # path, a one-gibibyte update buffer, one actor node with one GPU, retract pause mode) updated
    # by any keyword overrides.
    defaults = dict(
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["linear_qkv", "linear_proj"],
        megatron_to_hf_mode="bridge",
        rollout_num_gpus_per_engine=1,
        hf_checkpoint="/fake/path",
        update_weight_buffer_size=1 << 30,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        pause_generation_mode="retract",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# check_weight_sync_results
# ---------------------------------------------------------------------------


class TestCheckWeightSyncResults:
    # [CLASS DOC] Tests of the engine result checker.
    """Validate that check_weight_sync_results raises on engine failures."""

    def test_success_results_pass(self):
        # [FUNCTION DOC] A single successful result passes without raising.
        results = [_FakeEngineResult(success=True)]
        check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_lora(self):
        # [FUNCTION DOC] A failed result with the LoRA flag on raises a runtime error whose message
        # says the LoRA weight sync failed.
        results = [_FakeEngineResult(success=False, error_message="incompatible format")]
        with pytest.raises(RuntimeError, match="LoRA weight sync failed"):
            check_weight_sync_results(results, is_lora=True)

    def test_failure_result_raises_for_base(self):
        # [FUNCTION DOC] A failed result with the LoRA flag off raises a runtime error whose message
        # says the base model weight sync failed.
        results = [_FakeEngineResult(success=False, error_message="bad version")]
        with pytest.raises(RuntimeError, match="Base model weight sync failed"):
            check_weight_sync_results(results, is_lora=False)

    def test_plain_tuple_results_pass(self):
        # [FUNCTION DOC] A result that is a plain pair of true and a message (not a record) passes
        # without raising.
        """Non-dataclass results (e.g. (True, 'Success') tuples) should not raise."""
        results = [(True, "Success")]
        check_weight_sync_results(results, is_lora=False)

    def test_mixed_results_raises_on_first_failure(self):
        # [FUNCTION DOC] With a success followed by a failure carrying the message "oops", the
        # checker raises a runtime error containing that message.
        results = [
            _FakeEngineResult(success=True),
            _FakeEngineResult(success=False, error_message="oops"),
        ]
        with pytest.raises(RuntimeError, match="oops"):
            check_weight_sync_results(results, is_lora=True)


class TestUpdateWeightsEmptyBaseIteration:
    # [CLASS DOC] Test of the updater with an iterator that yields no buckets.
    def test_no_raise_for_base_model_zero_buckets(self):
        # [FUNCTION DOC] Builds a non-LoRA updater around a mock protocol (session support on, sender
        # role, begin sync returns true, one mock engine) and an iterator yielding nothing, with the
        # engine pause, begin, version, end and resume helpers, distributed module and gloo group
        # patched (rank zero). Runs one update and asserts: no bucket was sent, the protocol's
        # after-base-weights hook was called exactly once, and the updater's weight version is one.
        """Base model weight sync with zero buckets is valid (e.g. empty model state)."""
        empty_iterator = MagicMock()
        empty_iterator.iter_hf_weights.return_value = iter([])

        protocol = MagicMock()
        protocol.use_weight_update_session = True
        protocol.needs_base_resync_for_lora = False
        protocol.is_sender = True
        protocol.group_name = "test"
        protocol.begin_sync.return_value = True
        protocol.rollout_engines = [MagicMock()]

        args = _make_args(custom_model_provider_path=None, check_lora_weight_equal=False)
        with (
            patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol),
            patch(f"{_UPDATER_MODULE}.dist") as mock_dist,
            patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
            patch(f"{_UPDATER_MODULE}.pause_engines"),
            patch(f"{_UPDATER_MODULE}.begin_weight_update"),
            patch(f"{_UPDATER_MODULE}.set_weight_version"),
            patch(f"{_UPDATER_MODULE}.end_weight_update"),
            patch(f"{_UPDATER_MODULE}.resume_engines"),
        ):
            mock_dist.get_rank.return_value = 0
            updater = WeightUpdater(
                args,
                [MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen",
                quantization_config=None,
                iterator_factory=lambda *a, **k: empty_iterator,
                parallel_state=MagicMock(),
                is_lora=False,
            )
            updater.update_weights()

        protocol.send_bucket.assert_not_called()
        protocol.after_base_weights.assert_called_once()
        assert updater.weight_version == 1


# ---------------------------------------------------------------------------
# FlattenedTensorBucket round-trip correctness
# ---------------------------------------------------------------------------


class TestFlattenedTensorBucketRoundTrip:
    # [CLASS DOC] Round-trip tests of SGLang's flattened tensor bucket.
    """Verify serialize -> reconstruct preserves tensor values exactly."""

    def _get_bucket_class(self):
        # [FUNCTION DOC] HELPER: Imports and returns SGLang's flattened tensor bucket class; skips
        # the calling test if the import fails.
        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
        except ImportError:
            pytest.skip("sglang FlattenedTensorBucket not available")
        return FlattenedTensorBucket

    def test_roundtrip_single_dtype(self):
        # [FUNCTION DOC] Packs two random bfloat16 tensors into a bucket, reconstructs, and asserts
        # same count, names, shapes, dtypes and exactly equal values, in order.
        FlattenedTensorBucket = self._get_bucket_class()
        tensors = [
            ("a", torch.randn(4, 4, dtype=torch.bfloat16)),
            ("b", torch.randn(2, 8, dtype=torch.bfloat16)),
        ]

        bucket = FlattenedTensorBucket(named_tensors=tensors)
        reconstructed = bucket.reconstruct_tensors()

        assert len(reconstructed) == len(tensors)
        for (orig_name, orig_t), (rec_name, rec_t) in zip(tensors, reconstructed, strict=True):
            assert orig_name == rec_name
            assert orig_t.shape == rec_t.shape
            assert orig_t.dtype == rec_t.dtype
            assert torch.equal(orig_t, rec_t), f"Tensor {orig_name} values differ after round-trip"

    def test_roundtrip_mixed_dtypes(self):
        # [FUNCTION DOC] Packs bfloat16, float32 and float16 tensors into one bucket and asserts that
        # reconstruction raises a runtime error mentioning the storage offset, pinning the current
        # upstream behavior (the docstring explains the latent alignment problem). The test will
        # start failing when either side is fixed, which is the intended signal.
        """FIXME(sglang upstream contract): SGLang exposes
        ``FlattenedTensorBucket.supports_multi_dtypes = True`` but
        ``reconstruct_tensors()`` actually raises ``RuntimeError`` on mixed
        dtypes, because PyTorch ``view()`` requires ``storage_offset`` to be
        divisible by the target element size and concatenated flat buffers do
        not align across heterogeneous element sizes.

        This is a latent production landmine: ``_send_to_colocated_engine`` in
        ``miles/backends/training_utils/weight_update/protocols/cuda_ipc.py``
        reads the flag and packs mixed dtypes into a single bucket. In practice
        LoRA weights are uniform dtype, but FP8 / INT4 mixed-precision base
        weight sync would crash on sglang's receiver.

        Fix path (either side):
          - miles side: stop trusting ``supports_multi_dtypes`` in
            ``_send_to_colocated_engine`` and always group by dtype (matches
            the FSDP path's existing implementation in
            ``fsdp_utils/hf_weight_iterator.py``).
          - sglang side: actually align ``storage_offset`` in reconstruction.

        Until one side is fixed, this test asserts the current observed
        failure so we notice when either side changes.
        """
        FlattenedTensorBucket = self._get_bucket_class()

        tensors = [
            ("a_bf16", torch.randn(3, 3, dtype=torch.bfloat16)),
            ("b_fp32", torch.randn(2, 2, dtype=torch.float32)),
            ("c_fp16", torch.randn(5, dtype=torch.float16)),
        ]
        bucket = FlattenedTensorBucket(named_tensors=tensors)
        with pytest.raises(RuntimeError, match=r"storage_offset"):
            bucket.reconstruct_tensors()

    def test_roundtrip_from_flattened_data(self):
        # [FUNCTION DOC] Builds a sender bucket from two bfloat16 tensors, takes its flat tensor and
        # metadata, builds a receiver bucket from just those, reconstructs, and asserts names and
        # values equal the originals in order.
        """Simulate the receiver side: reconstruct from flattened_tensor + metadata."""
        FlattenedTensorBucket = self._get_bucket_class()

        original = [
            ("lora_A", torch.randn(8, 2, dtype=torch.bfloat16)),
            ("lora_B", torch.randn(2, 8, dtype=torch.bfloat16)),
        ]

        sender_bucket = FlattenedTensorBucket(named_tensors=original)
        flat_tensor = sender_bucket.get_flattened_tensor()
        metadata = sender_bucket.get_metadata()

        receiver_bucket = FlattenedTensorBucket(flattened_tensor=flat_tensor, metadata=metadata)
        reconstructed = receiver_bucket.reconstruct_tensors()

        for (orig_name, orig_t), (rec_name, rec_t) in zip(original, reconstructed, strict=True):
            assert orig_name == rec_name
            assert torch.equal(orig_t, rec_t)

    def test_lora_only_tensors_filtered_correctly(self):
        # [FUNCTION DOC] Filters a three-tensor list to the two LoRA tensors using the name
        # predicate, asserts two remain, packs and reconstructs them, and asserts names and values
        # round-trip exactly.
        """Verify that after filtering, only LoRA tensors survive and round-trip intact."""
        FlattenedTensorBucket = self._get_bucket_class()

        mixed = [
            ("model.layers.0.q_proj.weight", torch.randn(4, 4)),
            ("model.layers.0.q_proj.lora_A.weight", torch.randn(4, 2)),
            ("model.layers.0.q_proj.lora_B.weight", torch.randn(2, 4)),
        ]

        lora_only = [(n, t) for n, t in mixed if is_lora_weight_name(n)]
        assert len(lora_only) == 2

        bucket = FlattenedTensorBucket(named_tensors=lora_only)
        reconstructed = bucket.reconstruct_tensors()

        for (orig_name, orig_t), (rec_name, rec_t) in zip(lora_only, reconstructed, strict=True):
            assert orig_name == rec_name
            assert torch.equal(orig_t, rec_t)
