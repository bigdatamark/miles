# [MODULE DOC] Unit tests for flatten_inference_engine_checksums, which turns the per-engine
# response bodies of a weight-checksum request into one flat checksum dict per engine. Each rank
# of an engine is prefixed "rank<N>/" (N taken from the GPU rank in its parallelism info) and
# merged; engine bodies that are none are dropped; failures and malformed bodies raise assertion
# errors. Pure CPU tests on hand-built dictionaries.
"""Tests for engine_weight_checksum.flatten_inference_engine_checksums."""

from typing import Any

import pytest

from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums


def _engine_body(*, success: bool, ranks: list[dict[str, Any]] | None) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds a fake engine response body: a dict with the success flag and
    # the message "ok"; the "ranks" entry (the given list of per-rank dicts) is included only when
    # ranks is not none.
    body: dict[str, Any] = {"success": success, "message": "ok"}
    if ranks is not None:
        body["ranks"] = ranks
    return body


def _rank(rank: int, checksums: dict[str, str]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds one fake per-rank entry holding the checksums mapping and a
    # parallelism info list with a single "target" role entry carrying the given rank.
    return {"checksums": checksums, "parallelism_info": [{"role": "target", "rank": rank}]}


class TestFlattenInferenceEngineChecksums:
    # [CLASS DOC] Tests of the flatten function: input is a list of engine bodies (or none), output
    # a list of dicts mapping "rank<N>/<name>" to checksum strings.
    def test_single_engine_single_rank(self) -> None:
        # [FUNCTION DOC] One engine with one rank 0 yields a single dict with key "rank0/w".
        """One engine with one rank yields one prefixed checksum dict."""
        result = [_engine_body(success=True, ranks=[_rank(0, {"w": "aaa"})])]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "aaa"}]

    def test_multiple_engines_kept_in_order(self) -> None:
        # [FUNCTION DOC] Two engines produce two output dicts in the same order as the input.
        """Engines stay in their per-cell order."""
        result = [
            _engine_body(success=True, ranks=[_rank(0, {"w": "e0"})]),
            _engine_body(success=True, ranks=[_rank(0, {"w": "e1"})]),
        ]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "e0"}, {"rank0/w": "e1"}]

    def test_none_node_rank_payloads_filtered(self) -> None:
        # [FUNCTION DOC] A none entry (from a non-zero node rank) is dropped; output has only the
        # real engine's dict.
        """None engine bodies (non-zero node ranks) are dropped before indexing."""
        result = [_engine_body(success=True, ranks=[_rank(0, {"w": "e0"})]), None]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "e0"}]

    def test_all_none_fails_loud(self) -> None:
        # [FUNCTION DOC] An input of only none entries raises an assertion error matching "no
        # non-None engine bodies".
        """A None-only result means the checksum action did nothing, so fail loud."""
        result = [None, None]
        with pytest.raises(AssertionError, match="no non-None engine bodies"):
            flatten_inference_engine_checksums(result)

    def test_multi_rank_merged_with_rank_prefix(self) -> None:
        # [FUNCTION DOC] One engine with ranks 0 and 1 (same checksum name) yields one dict with both
        # "rank0/w" and "rank1/w", showing the prefix prevents key collisions.
        """Multiple ranks of one engine merge into one dict, prefixed by rank to avoid clobber."""
        result = [_engine_body(success=True, ranks=[_rank(0, {"w": "r0"}), _rank(1, {"w": "r1"})])]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "r0", "rank1/w": "r1"}]

    def test_ranks_out_of_order_sorted_by_parallelism_rank(self) -> None:
        # [FUNCTION DOC] Ranks listed as 1 then 0 give a result equal to the same ranks listed as 0
        # then 1 (order is normalized by parallelism rank).
        """Ranks arriving out of order (zmq) are sorted by parallelism rank deterministically."""
        out_of_order = [_engine_body(success=True, ranks=[_rank(1, {"w": "r1"}), _rank(0, {"w": "r0"})])]
        in_order = [_engine_body(success=True, ranks=[_rank(0, {"w": "r0"}), _rank(1, {"w": "r1"})])]
        assert flatten_inference_engine_checksums(out_of_order) == flatten_inference_engine_checksums(in_order)

    def test_engine_failure_fails_loud(self) -> None:
        # [FUNCTION DOC] A body with success false raises an assertion error matching "reported
        # failure", even though it has ranks.
        """An engine reporting success=False fails loud rather than silently dropping it."""
        result = [_engine_body(success=False, ranks=[_rank(0, {"w": "aaa"})])]
        with pytest.raises(AssertionError, match="reported failure"):
            flatten_inference_engine_checksums(result)

    def test_engine_without_ranks_fails_loud(self) -> None:
        # [FUNCTION DOC] A successful body lacking the ranks entry raises an assertion error
        # matching "no ranks".
        """A success body with no ranks fails loud (nothing to compare)."""
        result = [_engine_body(success=True, ranks=None)]
        with pytest.raises(AssertionError, match="no ranks"):
            flatten_inference_engine_checksums(result)

    def test_multi_role_parallelism_info_collapses_to_one_gpu_rank(self) -> None:
        # [FUNCTION DOC] A rank whose parallelism info lists "target" and "draft" roles both with
        # rank 0 collapses to the single GPU rank 0, giving key "rank0/w".
        """Per-role parallelism_info (target + draft) sharing a GPU rank uses that single rank."""
        rank_info = {
            "checksums": {"w": "aaa"},
            "parallelism_info": [{"role": "target", "rank": 0}, {"role": "draft", "rank": 0}],
        }
        result = [_engine_body(success=True, ranks=[rank_info])]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "aaa"}]

    def test_parallelism_info_rank_disagreement_fails_loud(self) -> None:
        # [FUNCTION DOC] Roles "target" rank 0 and "draft" rank 1 within one rank entry raise an
        # assertion error matching "one GPU rank".
        """Roles reporting different GPU ranks for one engine rank is a contract violation."""
        rank_info = {
            "checksums": {"w": "aaa"},
            "parallelism_info": [{"role": "target", "rank": 0}, {"role": "draft", "rank": 1}],
        }
        result = [_engine_body(success=True, ranks=[rank_info])]
        with pytest.raises(AssertionError, match="one GPU rank"):
            flatten_inference_engine_checksums(result)
