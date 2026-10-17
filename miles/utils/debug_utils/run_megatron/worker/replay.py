# [MODULE DOC] Mixture-of-experts routing replay for the standalone worker, built on the shared
# routing replay manager. A baseline run records the top expert indices of every replay object and
# saves them to one file named rank0_<manager filename> in the dump directory; a target run loads
# that file and slices each tensor for its own context-parallel (zigzag) and sequence-parallel
# layout. Only a single-process baseline is supported.
"""Routing replay stage management for standalone Megatron worker.

Only single-rank (nproc=1) baselines are supported: save always writes
a rank-0 file, and load always reads that file with CP/SP slicing.
"""

from pathlib import Path
from typing import NamedTuple

import torch

from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.replay_base import routing_replay_manager


class _ParallelRanks(NamedTuple):
    # [CLASS DOC] HELPER: Record of context-parallel size and rank and tensor-parallel size and
    # rank of the current process.
    cp_size: int
    cp_rank: int
    tp_size: int
    tp_rank: int


def setup_replay_before_model(script: WorkerScriptArgs) -> None:
    # [FUNCTION DOC] Mutates the global routing replay manager. If a dump path is set: enables the
    # manager and sets its stage to "record". Else if a load path is set: enables it and sets the
    # stage to "replay_forward". If both are set the dump path wins. If neither, does nothing.
    # Prints which mode was chosen.
    """Enable replay manager and set stage BEFORE model construction.

    Must be called before ``get_model()`` so that ``register_to_module``
    (called during model construction) sees ``enabled=True`` and creates
    Replay objects on MoE modules.
    """
    if script.routing_replay_dump_path:
        routing_replay_manager.enabled = True
        routing_replay_manager.stage = "record"
        print(f"[worker] Routing replay enabled, stage=record (dump → {script.routing_replay_dump_path})", flush=True)
    elif script.routing_replay_load_path:
        routing_replay_manager.enabled = True
        routing_replay_manager.stage = "replay_forward"
        print(
            f"[worker] Routing replay enabled, stage=replay_forward (load ← {script.routing_replay_load_path})",
            flush=True,
        )


def load_replay_data(
    script: WorkerScriptArgs,
    *,
    rank: int,
    sequence_parallel: bool = False,
) -> None:
    # [FUNCTION DOC] Does nothing when no load path is configured. Otherwise lists files in the
    # load directory matching "*_<manager filename>" in sorted order; raises a value error unless
    # exactly one is found (the message lists the names found). Then loads it into the manager's
    # replays with this rank and the sequence-parallel flag (default false). Mutates the manager's
    # replay objects.
    """Load routing replay data from rank 0's file with CP/SP slicing."""
    if not script.routing_replay_load_path:
        return

    load_dir: Path = script.routing_replay_load_path
    replay_files: list[Path] = sorted(load_dir.glob(f"*_{routing_replay_manager.filename}"))
    if len(replay_files) != 1:
        raise ValueError(
            f"Expected exactly 1 replay file in {load_dir}, "
            f"found {len(replay_files)}: {[f.name for f in replay_files]}"
        )

    replay_file: Path = replay_files[0]
    _load_replay(replay_file, rank=rank, sequence_parallel=sequence_parallel)


def save_replay_data(script: WorkerScriptArgs, *, rank: int) -> None:
    # [FUNCTION DOC] Does nothing when no dump path is configured. Otherwise asserts the rank is
    # zero, creates the dump directory, collects each replay's recorded list of top-index tensors
    # from the manager, asserts that at least one entry was recorded in total, and saves the list
    # (one list per replay) with the torch serializer to rank0_<manager filename> in the directory.
    # Prints entry and replay counts.
    """Save recorded routing replay data to disk (rank 0 only)."""
    if not script.routing_replay_dump_path:
        return
    assert rank == 0

    script.routing_replay_dump_path.mkdir(parents=True, exist_ok=True)

    replays_data: list[list[torch.Tensor]] = [replay.top_indices_list for replay in routing_replay_manager.replays]
    total_entries: int = sum(len(d) for d in replays_data)
    assert total_entries > 0

    save_path: Path = _replay_file_path(base_dir=script.routing_replay_dump_path)
    torch.save(replays_data, save_path)
    print(
        f"[worker] Saved routing replay ({total_entries} entries, {len(replays_data)} replays) → {save_path}",
        flush=True,
    )


def _replay_file_path(*, base_dir: Path) -> Path:
    # [FUNCTION DOC] HELPER: Path of the saved replay file: the directory joined with the name
    # "rank0_" followed by the manager's file name.
    return base_dir / f"rank0_{routing_replay_manager.filename}"


def _get_parallel_ranks() -> _ParallelRanks:
    # [FUNCTION DOC] HELPER: Imports Megatron and the parallel-state getter lazily. If Megatron's
    # model-parallel state is not initialized, returns size 1 and rank 0 for both context and
    # tensor parallelism; otherwise returns the sizes and ranks from the parallel state.
    """Return parallel ranks, defaulting to 1/0 if mpu is not initialized."""
    from megatron.core import mpu

    from miles.backends.training_utils.parallel import get_parallel_state

    if not mpu.is_initialized():
        return _ParallelRanks(cp_size=1, cp_rank=0, tp_size=1, tp_rank=0)

    state = get_parallel_state()
    return _ParallelRanks(
        cp_size=state.cp.size,
        cp_rank=state.cp.rank,
        tp_size=state.tp.size,
        tp_rank=state.tp.rank,
    )


def _load_replay(
    replay_file: Path,
    *,
    rank: int,
    sequence_parallel: bool,
) -> None:
    # [FUNCTION DOC] HELPER: Loads the saved list of per-replay tensor lists (full deserialization
    # allowed). Raises a value error if its length differs from the number of replay objects in
    # the manager. Sequence-parallel slicing applies only if the flag is set, the manager says
    # routing happens in a sequence-parallel region, and tensor-parallel size exceeds 1. For each
    # replay and its saved tensors: when context-parallel size exceeds 1, converts each tensor
    # from natural order to this rank's zigzag slice along dimension 0; when sequence-parallel
    # slicing applies, takes this tensor-parallel rank's contiguous chunk along dimension 0. Sets
    # the replay's index list to the result and resets its forward and backward indices to 0.
    # Rank 0 prints per-replay shapes before and after and a final summary.
    """Load replay from rank 0's file with CP zigzag slicing and SP slicing."""
    saved_replays: list[list[torch.Tensor]] = torch.load(replay_file, weights_only=False)

    expected: int = len(routing_replay_manager.replays)
    if len(saved_replays) != expected:
        raise ValueError(f"Replay file has {len(saved_replays)} replays but model expects {expected}")

    ranks: _ParallelRanks = _get_parallel_ranks()
    do_sp_slice: bool = sequence_parallel and routing_replay_manager.if_sp_region and ranks.tp_size > 1

    total_entries: int = 0
    for replay_idx, (replay, indices_list) in enumerate(
        zip(routing_replay_manager.replays, saved_replays, strict=True)
    ):
        sliced: list[torch.Tensor] = indices_list

        if ranks.cp_size > 1:
            from miles.backends.training_utils.data.context_parallel import natural_to_zigzag_slice

            sliced = [natural_to_zigzag_slice(t, dim=0, cp_size=ranks.cp_size, cp_rank=ranks.cp_rank) for t in sliced]

        if do_sp_slice:
            sliced = [_sp_slice(t, tp_size=ranks.tp_size, tp_rank=ranks.tp_rank) for t in sliced]

        replay.top_indices_list = sliced
        replay.forward_index = 0
        replay.backward_index = 0
        total_entries += len(sliced)

        if rank == 0:
            shapes_before: list[torch.Size] = [t.shape for t in indices_list]
            shapes_after: list[torch.Size] = [t.shape for t in sliced]
            print(
                f"[worker] replay[{replay_idx}]: cp={ranks.cp_size}/{ranks.cp_rank}, tp={ranks.tp_size}/{ranks.tp_rank}, "
                f"sp={sequence_parallel}, shapes {shapes_before} → {shapes_after}",
                flush=True,
            )

    if rank == 0:
        print(
            f"[worker] Loaded routing replay ({total_entries} entries, {expected} replays) ← {replay_file}",
            flush=True,
        )


def _sp_slice(tensor: torch.Tensor, *, tp_size: int, tp_rank: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Asserts the first dimension is divisible by the tensor-parallel size,
    # then returns the contiguous chunk of length (size divided by tp size) belonging to the rank
    # (a view).
    """Slice tensor along dim=0 for sequence parallelism."""
    seqlen: int = tensor.size(0)
    assert seqlen % tp_size == 0, f"seqlen {seqlen} not divisible by tp_size {tp_size}"
    chunk_size: int = seqlen // tp_size
    start: int = chunk_size * tp_rank
    end: int = chunk_size * (tp_rank + 1)
    return tensor[start:end]
