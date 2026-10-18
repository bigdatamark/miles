# [MODULE DOC]
# Unit tests for the rollout data conversion module: the dynamic global batch size computation,
# post-processing of rollout samples (flattening nested lists, trimming to a multiple of the global
# batch size, recording a dynamic batch size in metadata, and failing when too few samples remain),
# and validation that compacted rollout ids are annotated consistently. Samples and arguments come
# from the shared conftest helpers; everything is pure in-memory computation.
from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.rollout_data_conversion import (
    _compute_dynamic_global_batch_size,
    postprocess_rollout_data,
    validate_compact_rollout_ids,
)


class TestComputeDynamicGlobalBatchSize:
    # [CLASS DOC] Tests the dynamic global batch size: the sample count rounded down to a multiple
    # of the data parallel size, falling back to the data parallel size when that would be zero.
    def test_rounds_down_to_multiple_of_dp_size(self):
        # [FUNCTION DOC] 13 samples with data parallel size 4 give 12.
        args = make_args(global_batch_size=64)
        # 13 samples, dp_size=4 → floor(13/4)*4 = 12
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=13)
        assert gbs == 12

    def test_returns_num_samples_when_already_aligned(self):
        # [FUNCTION DOC] 16 samples with data parallel size 4 give 16.
        args = make_args(global_batch_size=64)
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=16)
        assert gbs == 16

    def test_falls_back_to_dp_size_when_below_dp_size(self):
        # [FUNCTION DOC] 2 samples with data parallel size 4 give 4 (the data parallel size).
        """When num_samples < dp_size, the rounded result is 0, which would
        produce a divide-by-zero downstream — fallback to dp_size."""
        args = make_args(global_batch_size=64)
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=2)
        assert gbs == 4


class TestPostprocessRolloutData:
    # [CLASS DOC] Tests post-processing of rollout data, which returns the (possibly trimmed and
    # flattened) samples plus a metadata mapping.
    def test_aligned_input_passes_through_unchanged(self):
        # [FUNCTION DOC] 8 samples with global batch size 4: all 8 returned, metadata empty.
        """Happy path: len(data) % global_batch_size == 0 → no trim, no metadata."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(8)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 8
        assert meta == {}

    def test_unaligned_input_is_trimmed_to_multiple(self):
        # [FUNCTION DOC] 11 samples with global batch size 4 and trimming enabled: 8 returned,
        # metadata empty.
        """len % gbs != 0 → trim down to multiple. Tail samples are dropped."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(11)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 8
        assert meta == {}

    def test_disable_rollout_trim_samples_keeps_unaligned_data(self):
        # [FUNCTION DOC] 11 samples with batch size 4 and trimming disabled: all 11 returned.
        """With trim disabled, leave length as-is even if not divisible."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=True, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(11)]
        out, _meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 11

    def test_raises_when_trim_would_produce_zero_samples(self):
        # [FUNCTION DOC] 5 samples with batch size 64 and trimming enabled: a value error whose
        # message contains "Not enough samples" is raised.
        """5 samples, gbs=64 → trim_len=0 → can't proceed."""
        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        with pytest.raises(ValueError, match="Not enough samples"):
            postprocess_rollout_data(args, [make_sample()] * 5, train_parallel_config={"dp_size": 1})

    def test_dynamic_batch_size_computed_and_recorded_in_metadata(self):
        # [FUNCTION DOC] 10 samples, configured batch size 64, dynamic batch size on, data parallel
        # size 2: the metadata records a dynamic global batch size of 10 and all 10 samples are kept.
        """When use_dynamic_global_batch_size=True, the function recomputes gbs
        from the actual sample count and records it in metadata."""
        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=True)
        data = [make_sample(index=i) for i in range(10)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 2})
        # Dynamic gbs = floor(10/2)*2 = 10
        assert meta["dynamic_global_batch_size"] == 10
        assert len(out) == 10

    def test_flattens_nested_list_of_lists(self):
        # [FUNCTION DOC] Input of two nested lists (2 and 1 samples) with trimming disabled returns
        # a flat list of 3 samples.
        """The function supports list[list[Sample]] input by flattening."""
        args = make_args(global_batch_size=2, disable_rollout_trim_samples=True, use_dynamic_global_batch_size=False)
        nested = [[make_sample(index=0), make_sample(index=1)], [make_sample(index=2)]]
        out, _meta = postprocess_rollout_data(args, nested, train_parallel_config={"dp_size": 1})
        assert len(out) == 3


class TestValidateRolloutIdAnnotated:
    # [CLASS DOC] Tests validation of rollout ids on compacted rollouts: only when samples are
    # nested three deep and a leaf has several siblings must they carry one shared rollout id.
    def test_flat_list_skips_validation(self):
        # [FUNCTION DOC] A flat list of samples without rollout ids passes (no error).
        """Depth-1 leaf (flat list[Sample]) is the default rollout shape — no rollout_id required."""
        validate_compact_rollout_ids([make_sample(index=i) for i in range(4)])

    def test_default_nested_shape_skips_validation(self):
        # [FUNCTION DOC] A two-deep nesting (prompt by rollout) without rollout ids passes.
        """list[list[Sample]] (prompt x rollout): leaf at depth 1 — no rollout_id required."""
        validate_compact_rollout_ids([[make_sample(index=0), make_sample(index=1)]])

    def test_compact_shape_requires_rollout_id(self):
        # [FUNCTION DOC] A three-deep nesting whose leaf holds two samples without rollout ids
        # raises an assertion error saying the rollout id is unset.
        """list[list[list[Sample]]]: leaf at depth 2 with >1 sibling requires rollout_id."""
        compact = [[[make_sample(index=0), make_sample(index=1)]]]
        with pytest.raises(AssertionError, match="rollout_id is unset"):
            validate_compact_rollout_ids(compact)

    def test_compact_shape_siblings_must_share_rollout_id(self):
        # [FUNCTION DOC] Two siblings with rollout ids 7 and 8 raise an assertion error saying
        # siblings must share the rollout id.
        a, b = make_sample(index=0), make_sample(index=1)
        a.rollout_id, b.rollout_id = 7, 8
        with pytest.raises(AssertionError, match="share rollout_id"):
            validate_compact_rollout_ids([[[a, b]]])

    def test_compact_shape_valid_when_annotated(self):
        # [FUNCTION DOC] Two siblings both with rollout id 7 pass validation.
        a, b = make_sample(index=0), make_sample(index=1)
        a.rollout_id = b.rollout_id = 7
        validate_compact_rollout_ids([[[a, b]]])

    def test_compact_singleton_leaf_needs_no_annotation(self):
        # [FUNCTION DOC] A three-deep nesting whose leaf has a single sample without a rollout id
        # passes validation.
        """A single-sample leaf at depth >= 2 is unambiguous — no rollout_id required."""
        validate_compact_rollout_ids([[[make_sample(index=0)]]])
