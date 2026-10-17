# [MODULE DOC]
# Naming conventions and parsing helpers for pools, cells and workers. A cell id is the pool id
# followed by a dash and a zero-padded five-digit cell index; a worker name is the cell id
# followed by a dash and a zero-padded five-digit index of the worker within the cell. Also holds
# the length limits and patterns that keep derived names valid as Kubernetes DNS labels,
# subdomains and port names. Pool ids may themselves contain dashes, so parsing splits from the
# right.
import re
from typing import NamedTuple

DNS_LABEL_PATTERN: str = r"[a-z0-9]([-a-z0-9]*[a-z0-9])?"
DNS_SUBDOMAIN_PATTERN: str = rf"{DNS_LABEL_PATTERN}(\.{DNS_LABEL_PATTERN})*"
NAME_INDEX_PAD_WIDTH: int = 5
_NAME_INDEX_LIMIT: int = 10**NAME_INDEX_PAD_WIDTH
POOL_NAME_MAX_LENGTH = 40
_LONGEST_ENGINE_POOL_ID_AROUND_THE_INSTANCE_ID = "inference-engine--99-99"
DEPLOY_INSTANCE_ID_MAX_LENGTH = POOL_NAME_MAX_LENGTH - len(_LONGEST_ENGINE_POOL_ID_AROUND_THE_INSTANCE_ID)
TRAINER_CONTROLLER_POOL_ID_PREFIX = "trainer-controller-"
TRAINER_ID_MAX_LENGTH = POOL_NAME_MAX_LENGTH - len(TRAINER_CONTROLLER_POOL_ID_PREFIX)
PORT_NAME_MAX_LENGTH = 15
PORT_NAME_PATTERN = r"^([0-9]+-)*[0-9]*[a-z][a-z0-9]*(-[a-z0-9]+)*$"
_PORT_NAME_SEPARATORS = re.compile(r"[^a-z0-9]+")


def compute_port_name(name: str) -> str:
    # [FUNCTION DOC]
    # Turn an arbitrary name into a valid Kubernetes container port name: lower-case it, replace
    # every run of characters other than lower-case letters and digits with one dash, strip dashes
    # at both ends, truncate to 15 characters, and strip a trailing dash left by the truncation.
    # Asserts that the result matches the port name pattern (needs at least one letter, no
    # consecutive dashes); the assertion message shows the original and shortened names.
    # Returns the cleaned name. Distinct inputs may collapse to the same output.
    cleaned = _PORT_NAME_SEPARATORS.sub("-", name.lower()).strip("-")[:PORT_NAME_MAX_LENGTH].rstrip("-")
    assert (
        re.fullmatch(PORT_NAME_PATTERN, cleaned) is not None
    ), f"port name {name!r} shortens to {cleaned!r}, which Kubernetes rejects: it must match {PORT_NAME_PATTERN}"
    return cleaned


class ParsedCellId(NamedTuple):
    # [CLASS DOC]
    # Result of splitting a cell id: the pool id and the integer cell index.
    pool_id: str
    cell_index: int


def format_name_index(index: int) -> str:
    # [FUNCTION DOC]
    # Render an index as a zero-padded decimal string of width five. Asserts the index is at least
    # zero and below one hundred thousand.
    assert 0 <= index < _NAME_INDEX_LIMIT, f"name index must be in [0, {_NAME_INDEX_LIMIT}), got {index}"
    return f"{index:0{NAME_INDEX_PAD_WIDTH}d}"


def parse_cell_id(cell_id: str) -> ParsedCellId:
    # [FUNCTION DOC]
    # Split a cell id at its last dash into pool id and integer cell index. Raises a value error
    # if there is no dash or the suffix is not an integer.
    pool_id, cell_index = cell_id.rsplit("-", maxsplit=1)
    return ParsedCellId(pool_id=pool_id, cell_index=int(cell_index))


def compute_cell_id(*, pool_id: str, cell_index: int) -> str:
    # [FUNCTION DOC]
    # Build a cell id: the pool id, a dash, and the padded cell index (see the padding helper;
    # asserts the index range).
    return f"{pool_id}-{format_name_index(cell_index)}"


# TODO refactor & move later
def compute_worker_name(*, pool_id: str, cell_index: int = 0, worker_in_cell_index: int = 0) -> str:
    # [FUNCTION DOC]
    # Build a worker name: the pool id, a dash, the padded cell index, a dash, and the padded
    # worker-in-cell index. Both indices default to zero. Asserts each index range.
    return f"{pool_id}-{format_name_index(cell_index)}-{format_name_index(worker_in_cell_index)}"


# TODO refactor & move later
def parse_worker_name(worker_name: str) -> tuple[str, int, int]:
    # [FUNCTION DOC]
    # Split a worker name from the right into pool id, cell index and worker-in-cell index and
    # return them as a tuple of string and two integers. Raises a value error if there are fewer
    # than two dashes or the last two segments are not integers.
    pool_id, cell_index, worker_in_cell_index = worker_name.rsplit("-", maxsplit=2)
    return pool_id, int(cell_index), int(worker_in_cell_index)


def cell_id_of_worker(worker_name: str) -> str:
    # [FUNCTION DOC]
    # Return the id of the cell a worker belongs to, by parsing the worker name and rebuilding the
    # cell id from its pool id and cell index. Errors as for the worker name parser.
    pool_id, cell_index, _worker_in_cell_index = parse_worker_name(worker_name)
    return compute_cell_id(pool_id=pool_id, cell_index=cell_index)


def _worker_name_of_cell(cell_id: str, *, worker_in_cell_index: int = 0) -> str:
    # [FUNCTION DOC] HELPER: Name of the worker at a given index (default zero) in the given cell,
    # built by parsing the cell id and composing the worker name.
    # KNOWN ISSUE: private and not referenced inside this module; possibly dead code.
    parsed = parse_cell_id(cell_id)
    return compute_worker_name(
        pool_id=parsed.pool_id, cell_index=parsed.cell_index, worker_in_cell_index=worker_in_cell_index
    )
