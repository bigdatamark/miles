# [MODULE DOC]
# Process-global parallelism description for the training backend. Holds a ParallelState record
# (group info for intra-/independent data parallel, context, tensor, pipeline, expert and expert
# tensor parallelism, plus meshes and flags) in a module-level variable with setter/getter used by
# all loss and data utilities. Data parallelism is either "intra" (normal) or "indep" (independent
# fault-tolerance cells); at most one of them may be non-trivial.
from dataclasses import dataclass, field
from enum import auto

from torch.distributed.device_mesh import DeviceMesh

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


from miles.utils.ft_utils.process_group_utils import GroupInfo, GroupsInfo

_parallel_state: "ParallelState | None" = None


def set_parallel_state(state: "ParallelState") -> None:
    # [FUNCTION DOC] Store the given state in the module-global slot (replacing any previous one).
    global _parallel_state
    _parallel_state = state


def get_parallel_state() -> "ParallelState":
    # [FUNCTION DOC] Return the global parallel state; raises an assertion error if it has not been
    # set.
    assert _parallel_state is not None, "ParallelState not initialized. Call set_parallel_state() first."
    return _parallel_state


class _DPMode(StrEnum):
    # [CLASS DOC] HELPER: String enum of the active data-parallel mode: intra (regular data
    # parallel group is the active one) or indep (independent data-parallel groups are active).
    INTRA = auto()
    INDEP = auto()


@dataclass
class ParallelState:
    # [CLASS DOC] Mutable record of group information (rank and size and process groups) per
    # parallel dimension: intra-DP, intra-DP x CP, CP, TP, PP, EP, expert-TP, independent DP;
    # optional expert data parallel (including CP replication) and TP x DP x CP groups; a mapping of
    # named device meshes; the context-parallel communication type (string or per-layer sequence;
    # "a2a" means Ulysses-style); whether this rank is on the last pipeline stage (default true);
    # virtual pipeline size (default 1) and microbatch group size per virtual stage (default none).
    """Core parallel state shared across all backends.
    Required by the general training utils.
    """

    intra_dp: GroupInfo
    intra_dp_cp: GroupInfo
    cp: GroupInfo
    tp: GroupInfo
    pp: GroupInfo
    ep: GroupInfo
    etp: GroupInfo
    indep_dp: GroupInfo
    # Megatron expert replicas, including context-parallel replication.
    edp: GroupInfo | None = None
    tp_dp_cp: GroupInfo | None = None
    meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    cp_comm_type: str | list[str] | tuple[str, ...] | None = None
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None

    @property
    def _dp_mode(self) -> _DPMode:
        # [FUNCTION DOC] HELPER: Property. A group is trivial if rank 0 and size 1. Asserts at least
        # one of intra-DP and independent DP is trivial; returns intra mode if the independent DP is
        # trivial, else independent mode.
        intra_trivial = self.intra_dp.rank == 0 and self.intra_dp.size == 1
        indep_trivial = self.indep_dp.rank == 0 and self.indep_dp.size == 1
        assert intra_trivial or indep_trivial, "intra_dp and indep_dp cannot both be non-trivial"

        return _DPMode.INTRA if indep_trivial else _DPMode.INDEP

    @property
    def effective_dp(self) -> GroupInfo:
        # [FUNCTION DOC] Property: the intra-DP group info in intra mode, else the independent DP
        # group info. Asserts via the mode helper.
        return {_DPMode.INTRA: self.intra_dp, _DPMode.INDEP: self.indep_dp}[self._dp_mode]

    @property
    def effective_dp_cp(self) -> GroupsInfo:
        # [FUNCTION DOC] Property: multi-level group info for gathering over DP x CP. In intra mode a
        # single-level wrapper of the intra-DP x CP group; in independent mode a two-level object with
        # the intra-DP x CP group as inner and the independent DP group as outer.
        return {
            _DPMode.INTRA: GroupsInfo.from_single(self.intra_dp_cp),
            _DPMode.INDEP: GroupsInfo.from_pair(inner=self.intra_dp_cp, outer=self.indep_dp),
        }[self._dp_mode]

    @property
    def is_metrics_rank(self) -> bool:
        # [FUNCTION DOC] Property: true when the effective DP x CP rank is 0, the TP rank is 0 and
        # this is the last pipeline stage, i.e. the single rank that reports metrics.
        return self.effective_dp_cp.rank == 0 and self.tp.rank == 0 and self.is_pp_last_stage

    @property
    def is_ulysses_cp(self) -> bool:
        # [FUNCTION DOC] Property: true when CP size > 1 and the CP communication type is "a2a"
        # (all-to-all); if the type is a list/tuple, the first element is used (none if empty).
        cp_comm_type = self.cp_comm_type
        if isinstance(cp_comm_type, (list, tuple)):
            cp_comm_type = cp_comm_type[0] if cp_comm_type else None
        return self.cp.size > 1 and cp_comm_type == "a2a"

    def get_mesh(self, name: str) -> DeviceMesh:
        # [FUNCTION DOC] Look up a named device mesh; raises a key error if absent.
        return self.meshes[name]
