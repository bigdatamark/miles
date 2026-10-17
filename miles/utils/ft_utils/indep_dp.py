# [MODULE DOC]
# Information for "independent data parallelism" in fault tolerant training, where several cells
# (replicas) train independently and only the currently alive ones synchronize. Defines the
# immutable description of this cell's position among alive cells and a helper that creates a tcp
# store for coordination. Torch is only imported lazily (type checking and inside the helper).
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class IndepDPInfo:
    # [CLASS DOC] Immutable view of the independent data parallel group: this cell's index among all
    # cells, the total cells, this cell's rank among alive cells, the count of alive cells, the
    # quorum id (identifies the current set of alive cells), and the sorted-by-convention list of
    # alive cell indices. Validated after construction: the alive rank must equal the position of
    # this cell index within the alive list, and the alive size must equal the list length.
    cell_index: int
    num_cells: int
    alive_rank: int
    alive_size: int
    quorum_id: int
    alive_cell_indices: list[int]

    @classmethod
    def create_trivial(cls) -> "IndepDPInfo":
        # [FUNCTION DOC] Info for a single cell world: cell index 0 of 1, alive rank 0 of 1, quorum
        # id 0, alive cells list holding just zero.
        return cls(
            cell_index=0,
            num_cells=1,
            alive_rank=0,
            alive_size=1,
            quorum_id=0,
            alive_cell_indices=[0],
        )

    def __post_init__(self):
        # [FUNCTION DOC] HELPER: Validation. Asserts alive rank equals the index of this cell in the
        # alive list (a value error from the list search if the cell is absent) and alive size equals
        # the list length. Does not check cell index range against the cell count.
        assert self.alive_rank == self.alive_cell_indices.index(self.cell_index)
        assert self.alive_size == len(self.alive_cell_indices)


def create_tcp_store() -> tuple["torch.distributed.TCPStore", str]:
    # [FUNCTION DOC] Creates a master tcp store listening on all interfaces at an OS-chosen free
    # port, without waiting for workers. Returns the store and its address text "host:port" where the
    # host is this node's ip (not the wildcard address). Imports torch and the node ip helper
    # lazily to keep module import light. Side effect: opens a listening socket owned by the
    # returned store.
    import torch.distributed

    from miles.utils.misc import get_current_node_ip

    store = torch.distributed.TCPStore(
        host_name="0.0.0.0",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    host = get_current_node_ip()
    port = store.port
    return store, f"{host}:{port}"
