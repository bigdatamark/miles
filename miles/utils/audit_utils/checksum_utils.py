# [MODULE DOC] Normalizes the reply of the inference engines' weight-check call in checksum mode
# into one flat mapping per engine. The alias InferenceEngineChecksums is a mapping from a
# rank-prefixed tensor name to its checksum string. Pure functions; failures are assertions.

from typing import Any

InferenceEngineChecksums = dict[str, str]


def flatten_inference_engine_checksums(check_weights_result: Any) -> list[InferenceEngineChecksums]:
    # [FUNCTION DOC] Input: an iterable with one reply body per engine (None for engines that did
    # not answer). Drops None bodies, asserts at least one remains, and returns one merged
    # checksum mapping per remaining engine, in the original order. Asserts propagate from the
    # merge for malformed bodies.
    engine_bodies = list(check_weights_result)
    surviving = [body for body in engine_bodies if body is not None]
    assert surviving, (
        f"check_weights('checksum') returned no non-None engine bodies "
        f"(got {len(engine_bodies)} entries, all None): {check_weights_result!r}"
    )
    return [_merge_inference_engine_ranks(body) for body in surviving]


def _merge_inference_engine_ranks(engine_body: dict[str, Any]) -> InferenceEngineChecksums:
    # [FUNCTION DOC] HELPER: Merges one engine's per-rank checksum mappings. Asserts the body's
    # success flag is true (default false when absent) and that it lists at least one rank. Ranks
    # are sorted by GPU rank, and each tensor name is prefixed with "rank" plus the rank number
    # and a slash, so identically named shards from different ranks cannot overwrite each other.
    # Returns the merged mapping.
    # Ranks arrive in non-deterministic (zmq) order under TP>1; sort and prefix each tensor
    # name with rank{r}/ so distinct shards' identically-named tensors never clobber.
    assert engine_body.get("success", False), f"check_weights engine reported failure: {engine_body!r}"
    ranks: list[dict[str, Any]] = engine_body.get("ranks", []) or []
    assert ranks, f"check_weights engine body has no ranks: {engine_body!r}"

    ranks_sorted = sorted(ranks, key=_gpu_rank)

    merged: InferenceEngineChecksums = {}
    for rank_info in ranks_sorted:
        rank = _gpu_rank(rank_info)
        for name, value in rank_info["checksums"].items():
            merged[f"rank{rank}/{name}"] = value
    return merged


def _gpu_rank(rank_info: dict[str, Any]) -> int:
    # [FUNCTION DOC] HELPER: Reads the parallelism info list of one rank entry (each element has
    # a rank value for one parallel role), asserts all roles agree on a single GPU rank, and
    # returns it. Raises a key error if the parallelism info is missing.
    parallelism_info = rank_info["parallelism_info"]
    gpu_ranks = {role_info["rank"] for role_info in parallelism_info}
    assert len(gpu_ranks) == 1, f"expected one GPU rank across roles, got {gpu_ranks}: {rank_info!r}"
    return next(iter(gpu_ranks))
