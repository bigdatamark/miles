# [MODULE DOC]
# Data model of the v2 session: an append-only forest of trajectory nodes (each node is the end
# of one model generation holding the messages it added, the full token snapshot from the root,
# the span of its sampled completion, commit order, record and finish reason), an attach-point
# result type, and a tree class with node creation (capped) and a search for the deepest node a
# request's messages extend. The commit sequence number is the only ordering key; wall-clock
# time is decoration only. Pure synchronous data; serving policy lives in session_state.
# Module constant: the maximum number of nodes per tree (1024).
"""Session-as-forest data model: trajectory nodes and attach-point search.

The forest is append-only and ``seq`` (commit order) is the only ordering
key. Everything here is synchronous pure data — serving policy,
concurrency, and tokenization live one layer up in ``session_state``.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.message_matcher_hub import SessionMessageMatcher, strict_message_matches

MAX_NODES = 1024


@dataclass
class TrajectoryNode:
    # [CLASS DOC]
    # One generation in the tree. Fields: messages this node added (the request messages beyond the
    # parent's path plus the assistant reply); the full token list from the root through this
    # node's completion; the (start, end) span of the sampled completion inside that list; the
    # commit sequence number; wall-clock commit time (decoration only); the upstream response id;
    # the recorded exchange; the finish reason; the full resolved request (own copy); parent
    # (none for a root); child nodes in creation order.
    """End of one model generation (SessionRecord is 1:1 with the node)."""

    delta_messages: list[dict[str, Any]]
    token_ids: list[int]  # full root->node snapshot
    completion_span: tuple[int, int]  # this node's sampled completion within token_ids
    seq: int  # per-session logical commit order — THE ordering key
    committed_at: float  # wall clock, decoration only (NTP-unsafe; never order by this)
    response_id: str  # upstream response id: the agent-branch <-> leaf join key
    record: SessionRecord
    finish_reason: str
    # Full resolved request for this generation, isolated from mutable request data.
    turn_args: dict[str, Any] = field(default_factory=dict)
    parent: "TrajectoryNode | None" = None
    children: list["TrajectoryNode"] = field(default_factory=list, repr=False)

    @property
    def truncated(self) -> bool:
        # [FUNCTION DOC] Property: true when the generation stopped because of the length limit.
        return self.finish_reason == "length"

    def path_nodes(self) -> list["TrajectoryNode"]:
        # [FUNCTION DOC] Nodes from the root down to this node inclusive, in order (walks the
        # parent links and reverses).
        nodes: list[TrajectoryNode] = []
        node: TrajectoryNode | None = self
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def path_messages(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC] The full message history to this node: the concatenation of the added
        # messages of every node on the path, root first.
        return [message for node in self.path_nodes() for message in node.delta_messages]


@dataclass(frozen=True)
class AttachPoint:
    # [CLASS DOC] Immutable result of the attach search: the node to attach under (none means a
    # new root), how many request messages that node's path consumed, and a diagnostic of the
    # deepest message overlap seen anywhere, including partially matching nodes.
    """Result of matching a request against the tree."""

    node: "TrajectoryNode | None"  # None = new root
    matched_messages: int  # request messages consumed by the attach node's path
    best_overlap: int  # diagnostics only: deepest overlap seen, incl. partial deltas


class SessionTree:
    # [CLASS DOC] Append-only forest: the list of root nodes and the list of all nodes in creation
    # order (a node's position equals its sequence number). Nodes are never removed.
    """Forest of trajectory nodes plus the append-only commit surface."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Start with no roots and no nodes.
        self.roots: list[TrajectoryNode] = []
        self.nodes: list[TrajectoryNode] = []  # creation (seq) order

    def leaves(self) -> list[TrajectoryNode]:
        # [FUNCTION DOC] All nodes without children, in creation order.
        return [node for node in self.nodes if not node.children]

    def create_node(
        self,
        parent: TrajectoryNode | None,
        *,
        delta_messages: list[dict[str, Any]],
        token_ids: list[int],
        completion_span: tuple[int, int],
        committed_at: float,
        response_id: str,
        record: SessionRecord,
        finish_reason: str,
        turn_args: dict[str, Any] | None = None,
    ) -> TrajectoryNode:
        # [FUNCTION DOC] Append a new node under the parent (none for a new root) and return it.
        # Raises a value error if the tree already holds the maximum number of nodes (which
        # usually means the client is not replaying history verbatim). The node copies the delta
        # messages and token ids lists, deep-copies the turn args (empty mapping if none), takes
        # sequence number equal to the current node count, and is registered in the node list and
        # in the roots list or its parent's children.
        if len(self.nodes) >= MAX_NODES:
            raise ValueError(
                f"node cap reached ({MAX_NODES}): the session cannot branch or extend "
                f"further — this almost always means the harness is not replaying "
                f"history verbatim"
            )
        node = TrajectoryNode(
            delta_messages=list(delta_messages),
            token_ids=list(token_ids),
            completion_span=completion_span,
            seq=len(self.nodes),
            committed_at=committed_at,
            response_id=response_id,
            record=record,
            finish_reason=finish_reason,
            turn_args=deepcopy(turn_args or {}),
            parent=parent,
        )
        self.nodes.append(node)
        if parent is None:
            self.roots.append(node)
        else:
            parent.children.append(node)
        return node

    def find_attach_point(
        self,
        request_messages: list[dict[str, Any]],
        *,
        message_matcher: SessionMessageMatcher | None = None,
    ) -> AttachPoint:
        # [FUNCTION DOC]
        # Depth-first search (explicit stack, roots and children visited in creation order) over
        # the forest without mutating it. For each visited node with the offset of request
        # messages consumed by its ancestors, count how many of its added messages match the
        # request messages starting at the offset (using the matcher, strict by default); track the
        # best overlap diagnostic as the largest offset plus matched count seen. A node with only a
        # partial match is not a candidate and its subtree is not searched. A fully matched node
        # becomes the best candidate if it consumes more request messages than the current best, or
        # equally many but has a larger sequence number; its children are then searched with the
        # new offset. Returns the best node with the consumed count, or no node (count zero) when no
        # root fully matches.
        """Deepest node whose full path messages are a prefix of the request.

        Message equivalence is decided by *message_matcher* (defaults to the
        strict matcher).  A node is only entered after its parent's delta is
        fully consumed; ties on depth (twins whose deltas both match) go to
        the latest ``seq``.  Pure judgment — never mutates the forest.
        """
        matcher = message_matcher if message_matcher is not None else strict_message_matches
        best: TrajectoryNode | None = None
        best_matched = -1
        best_overlap = 0

        stack = [(root, 0) for root in reversed(self.roots)]
        while stack:
            node, offset = stack.pop()
            delta = node.delta_messages
            i = 0
            while (
                i < len(delta)
                and offset + i < len(request_messages)
                and matcher(delta[i], request_messages[offset + i])
            ):
                i += 1
            best_overlap = max(best_overlap, offset + i)
            if i < len(delta):
                continue  # partial delta: this node (and its subtree) is not a candidate
            matched = offset + len(delta)
            if matched > best_matched or (matched == best_matched and best is not None and node.seq > best.seq):
                best, best_matched = node, matched
            stack.extend((child, matched) for child in reversed(node.children))

        if best is None:
            return AttachPoint(node=None, matched_messages=0, best_overlap=best_overlap)
        return AttachPoint(node=best, matched_messages=best_matched, best_overlap=best_overlap)
