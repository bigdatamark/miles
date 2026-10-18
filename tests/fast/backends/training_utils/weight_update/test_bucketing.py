# [MODULE DOC]
# CPU unit tests for the Hugging Face weight iterator bucketing helpers: assembling atomic update
# groups (sets of parameters, identified by name suffix per prefix instance, that must be sent in
# one unit) and packing units into size-bounded buckets without splitting any unit. A "unit" is a
# list of name and tensor pairs; tests use small zero-filled byte tensors. Registered for the CPU
# stage-a suite.
"""Unit tests for HF-namespace atomic grouping and size packing."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


import pytest
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import (
    AtomicUpdateGroup,
    assemble_atomic_update_groups,
    pack_units_by_size,
)


def _unit(*names, numel=4):
    # [FUNCTION DOC] HELPER: Builds one unit: a list of name and tensor pairs, one per given name,
    # each tensor a zero-filled one-dimensional byte tensor with numel elements (default four).
    return [(name, torch.zeros(numel, dtype=torch.uint8)) for name in names]


PAIR_GROUP = AtomicUpdateGroup("pair", (".b.weight", ".c.weight"))


class TestAssembleAtomicUpdateGroups:
    # [CLASS DOC] Tests of the atomic update group assembler. Shared fixture: a required group keyed
    # "pair" with the suffixes for the b and c weights. Results are compared as lists of name lists.
    def test_ungrouped_units_stream_through(self):
        # [FUNCTION DOC] Units whose names match no group suffix (with only an optional group
        # configured) are emitted unchanged, in order, as separate units.
        optional = AtomicUpdateGroup("pair", (".b.weight", ".c.weight"), optional=True)
        units = [_unit("layers.0.a.weight"), _unit("layers.1.a.weight")]
        out = list(assemble_atomic_update_groups(units, [optional]))
        assert [[n for n, _ in u] for u in out] == [["layers.0.a.weight"], ["layers.1.a.weight"]]

    def test_group_members_merge_into_one_unit(self):
        # [FUNCTION DOC] With units for a, b and c of layer zero, the b and c units are merged into one
        # unit (in arrival order) while the unrelated a unit passes through first and alone.
        units = [_unit("layers.0.a.weight"), _unit("layers.0.b.weight"), _unit("layers.0.c.weight")]
        out = list(assemble_atomic_update_groups(units, [PAIR_GROUP]))
        assert [[n for n, _ in u] for u in out] == [
            ["layers.0.a.weight"],
            ["layers.0.b.weight", "layers.0.c.weight"],
        ]

    def test_merge_is_per_prefix_instance(self):
        # [FUNCTION DOC] Interleaved members of layers zero and one are merged per layer prefix; each
        # merged unit is emitted when its group completes, so layer one (completed first) precedes
        # layer zero.
        units = [
            _unit("layers.0.b.weight"),
            _unit("layers.1.b.weight"),
            _unit("layers.1.c.weight"),
            _unit("layers.0.c.weight"),
        ]
        out = list(assemble_atomic_update_groups(units, [PAIR_GROUP]))
        assert [[n for n, _ in u] for u in out] == [
            ["layers.1.b.weight", "layers.1.c.weight"],
            ["layers.0.b.weight", "layers.0.c.weight"],
        ]

    def test_quant_scales_ride_inside_the_matching_unit(self):
        # [FUNCTION DOC] Units that also hold quantization scale tensors (names with an inverse scale
        # suffix) are matched through their weight member; the scale entries travel inside, so the
        # merged unit holds b weight, b scale, c weight, c scale in order.
        """A unit is matched by any member; non-matching members (scales) come along."""
        units = [
            _unit("layers.0.b.weight", "layers.0.b.weight_scale_inv"),
            _unit("layers.0.c.weight", "layers.0.c.weight_scale_inv"),
        ]
        out = list(assemble_atomic_update_groups(units, [PAIR_GROUP]))
        assert [[n for n, _ in u] for u in out] == [
            [
                "layers.0.b.weight",
                "layers.0.b.weight_scale_inv",
                "layers.0.c.weight",
                "layers.0.c.weight_scale_inv",
            ]
        ]

    def test_empty_units_are_skipped(self):
        # [FUNCTION DOC] An empty unit in the stream produces no output unit (no groups configured).
        out = list(assemble_atomic_update_groups([[], _unit("layers.0.a.weight")], []))
        assert [[n for n, _ in u] for u in out] == [["layers.0.a.weight"]]

    def test_incomplete_group_raises_at_end_of_stream(self):
        # [FUNCTION DOC] Only the b member arrives; once the stream is exhausted the assembler raises
        # an assertion error mentioning incomplete atomic update groups.
        with pytest.raises(AssertionError, match="Incomplete atomic update groups"):
            list(assemble_atomic_update_groups([_unit("layers.0.b.weight")], [PAIR_GROUP]))

    def test_required_group_matching_nothing_raises(self):
        # [FUNCTION DOC] A required group that matches no parameter in the whole stream raises an
        # assertion error saying it matched no params.
        with pytest.raises(AssertionError, match="matched no params"):
            list(assemble_atomic_update_groups([_unit("layers.0.a.weight")], [PAIR_GROUP]))

    def test_optional_group_matching_nothing_passes(self):
        # [FUNCTION DOC] The same situation with an optional group does not raise; the single
        # unrelated unit is emitted.
        optional = AtomicUpdateGroup("pair", (".b.weight", ".c.weight"), optional=True)
        out = list(assemble_atomic_update_groups([_unit("layers.0.a.weight")], [optional]))
        assert len(out) == 1

    def test_unit_matching_two_suffixes_raises(self):
        # [FUNCTION DOC] A single unit containing both the b and c weights (two suffixes of one
        # group) is ambiguous and raises an assertion error saying it matches multiple suffixes.
        with pytest.raises(AssertionError, match="matches multiple"):
            list(assemble_atomic_update_groups([_unit("layers.0.b.weight", "layers.0.c.weight")], [PAIR_GROUP]))

    def test_invalid_group_specs_raise(self):
        # [FUNCTION DOC] Even with an empty stream, invalid group specifications are rejected with
        # assertion errors: a group with no suffixes, a group with duplicate suffixes, and two groups
        # sharing the same key.
        for groups, error in [
            ([AtomicUpdateGroup("empty", ())], "has no suffixes"),
            ([AtomicUpdateGroup("dup", (".a", ".a"))], "duplicate suffixes"),
            ([AtomicUpdateGroup("k", (".a",)), AtomicUpdateGroup("k", (".b",))], "Duplicate atomic update group keys"),
        ]:
            with pytest.raises(AssertionError, match=error):
                list(assemble_atomic_update_groups([], groups))


class TestPackUnitsBySize:
    # [CLASS DOC] Tests of the size-based unit packer; results are compared as lists of name lists.
    def test_packs_up_to_max_bytes(self):
        # [FUNCTION DOC] Three four-byte units with a nine-byte limit pack into one bucket of the
        # first two and a second bucket with the third.
        units = [_unit("a", numel=4), _unit("b", numel=4), _unit("c", numel=4)]
        out = list(pack_units_by_size(units, max_bytes=9))
        assert [[n for n, _ in u] for u in out] == [["a", "b"], ["c"]]

    def test_never_splits_a_unit(self):
        # [FUNCTION DOC] A two-tensor unit (eight bytes) exceeding the six-byte limit is emitted whole
        # in its own bucket, after the preceding small unit's bucket; units are never split.
        units = [_unit("a", numel=4), _unit("b1", "b2", numel=4)]
        out = list(pack_units_by_size(units, max_bytes=6))
        assert [[n for n, _ in u] for u in out] == [["a"], ["b1", "b2"]]

    def test_empty_stream_yields_nothing(self):
        # [FUNCTION DOC] An empty input stream produces no buckets.
        assert list(pack_units_by_size([], max_bytes=8)) == []
