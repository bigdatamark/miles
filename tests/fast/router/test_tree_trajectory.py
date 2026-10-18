# [MODULE DOC] Pure model-level unit tests for the session trajectory tree: node creation and
# helpers (path nodes, path messages, token snapshots, truncation flag, leaf order, node cap,
# creation-order sequence numbers) and the attach-point search that decides where a new request's
# messages attach (strict extension, full resend, divergence in internal or root deltas, foreign
# assistant rewrites, zero overlap, twin ties, deepest match across roots, empty request,
# purity, and very deep trees beyond the recursion limit). Module constants are shared message
# dictionaries: system, user one, assistant one, tool one, assistant two, tool two, assistant
# three.
"""Unit tests for the trajectory tree: data model + attach-point search.

Pure model level (no serving wiring).
"""

import sys

import pytest

from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2 import tree_trajectory
from miles.rollout.session.v2.tree_trajectory import SessionTree

SYS = {"role": "system", "content": "sys"}
U1 = {"role": "user", "content": "q1"}
A1 = {"role": "assistant", "content": "a1"}
T1 = {"role": "tool", "content": "t1", "tool_call_id": "c1"}
A2 = {"role": "assistant", "content": "a2"}
T2 = {"role": "tool", "content": "t2", "tool_call_id": "c2"}
A3 = {"role": "assistant", "content": "a3"}


def _commit(tree, parent, delta, *, finish_reason="stop", committed_at=None):
    # [FUNCTION DOC] HELPER: Creates a node in the tree under the given parent (or a root). The
    # sequence is the current node count; token ids are the parent's ids plus one new id equal to
    # 100 plus the sequence, with the completion span covering that one new token. The commit
    # time defaults to the sequence as a float, the response id is derived from the sequence, and
    # the record is a successful chat-completions post with empty request and response. Returns
    # the node.
    seq = len(tree.nodes)
    prefix = parent.token_ids if parent is not None else []
    tokens = prefix + [100 + seq]
    record = SessionRecord(
        timestamp=0.0, method="POST", path="/v1/chat/completions", request={}, response={}, status_code=200
    )
    return tree.create_node(
        parent,
        delta_messages=delta,
        token_ids=tokens,
        completion_span=(len(tokens) - 1, len(tokens)),
        committed_at=committed_at if committed_at is not None else float(seq),
        response_id=f"resp-{seq}",
        record=record,
        finish_reason=finish_reason,
    )


@pytest.fixture
def linear_tree():
    # [FUNCTION DOC] HELPER: Fixture building a three-node chain (root with system, user,
    # assistant; then tool plus assistant; then tool plus assistant) and returning the tree and
    # the three nodes.
    """root n0=[sys,u1,a1] -> n1=[t1,a2] -> n2=[t2,a3]"""
    tree = SessionTree()
    n0 = _commit(tree, None, [SYS, U1, A1])
    n1 = _commit(tree, n0, [T1, A2])
    n2 = _commit(tree, n1, [T2, A3])
    return tree, n0, n1, n2


class TestAttachPoint:
    # [CLASS DOC] Tests of the tree's attach-point search; each result exposes the attach node, the
    # number of matched messages and the best overlap seen.
    def test_case1_strict_leaf_extension(self, linear_tree):
        # [FUNCTION DOC] A request extending the full path attaches at the last node with seven
        # matched messages.
        tree, _, _, n2 = linear_tree
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2, A3, {"role": "user", "content": "next"}])
        assert ap.node is n2
        assert ap.matched_messages == 7

    def test_case2_degenerate_resend_of_full_path(self, linear_tree):
        # [FUNCTION DOC] Resending exactly the full path attaches at the last node with seven
        # matched messages (empty suffix).
        tree, _, _, n2 = linear_tree
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2, A3])
        assert ap.node is n2
        assert ap.matched_messages == 7  # empty suffix: generate a new child of n2

    def test_case3_internal_node_divergence(self, linear_tree):
        # [FUNCTION DOC] A request diverging at the last tool message attaches at the middle node
        # with five matched messages and best overlap five.
        tree, _, n1, _ = linear_tree
        t2_diff = {"role": "tool", "content": "t2-DIFF", "tool_call_id": "c2"}
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, t2_diff])
        assert ap.node is n1
        assert ap.matched_messages == 5
        assert ap.best_overlap == 5

    def test_case4_divergence_inside_child_delta_attaches_parent(self, linear_tree):
        # [FUNCTION DOC] Diverging inside the second node's delta attaches at the root node with
        # three matched messages.
        tree, n0, _, _ = linear_tree
        t1_diff = {"role": "tool", "content": "t1-DIFF", "tool_call_id": "c1"}
        ap = tree.find_attach_point([SYS, U1, A1, t1_diff])
        assert ap.node is n0
        assert ap.matched_messages == 3

    def test_case4b_divergence_inside_root_delta_is_new_root(self, linear_tree):
        # [FUNCTION DOC] Diverging inside the root delta gives no attach node (new root), zero
        # matched messages and best overlap one (only the system message matched).
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([SYS, {"role": "user", "content": "other"}])
        assert ap.node is None
        assert ap.matched_messages == 0
        assert ap.best_overlap == 1  # sys matched before the divergence

    def test_case4b_pure_prefix_of_root_delta_is_new_root(self, linear_tree):
        # [FUNCTION DOC] A request that is only a proper prefix of the root delta (system, user)
        # gives no attach node and best overlap two.
        """Today's no-anchor shape: resend [sys, u1] against stored [sys, u1, a1, ...]."""
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([SYS, U1])
        assert ap.node is None
        assert ap.best_overlap == 2

    def test_case5_foreign_assistant_divergence_attaches_parent(self, linear_tree):
        # [FUNCTION DOC] A request whose second assistant was rewritten (compaction) attaches at
        # the root node with three matched messages and best overlap four (the tool message inside
        # the second node's delta also matched).
        """Client rewrote a2 (compaction): the rewritten assistant lands in the
        new branch's delta; matching just stops before it."""
        tree, n0, _, _ = linear_tree
        a2_rewritten = {"role": "assistant", "content": "a2-compacted"}
        ap = tree.find_attach_point([SYS, U1, A1, T1, a2_rewritten, {"role": "user", "content": "go on"}])
        assert ap.node is n0
        assert ap.matched_messages == 3
        assert ap.best_overlap == 4  # sys,u1,a1 + t1 inside n1's delta

    def test_case6_zero_overlap_is_new_root(self, linear_tree):
        # [FUNCTION DOC] A request sharing nothing with the tree (different system message) gives
        # no attach node and zero overlap.
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([{"role": "system", "content": "subagent"}, {"role": "user", "content": "task"}])
        assert ap.node is None
        assert ap.best_overlap == 0

    def test_case9_twin_tie_goes_to_latest_seq(self):
        # [FUNCTION DOC] Two children with identical delta text: a request extending them attaches
        # at the later one (larger sequence).
        tree = SessionTree()
        n0 = _commit(tree, None, [SYS, U1, A1])
        twin_a = _commit(tree, n0, [T1, A2])
        twin_b = _commit(tree, n0, [T1, A2])  # same delta text, later seq
        assert twin_a.seq < twin_b.seq
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, {"role": "user", "content": "next"}])
        assert ap.node is twin_b

    def test_deepest_match_wins_across_roots(self):
        # [FUNCTION DOC] With two roots sharing the opening messages, a request following the first
        # root's longer history attaches at that root's child, not at the other root.
        """Two roots where one's opening is a prefix of the other's history."""
        tree = SessionTree()
        r1 = _commit(tree, None, [SYS, U1, A1])
        n1 = _commit(tree, r1, [T1, A2])
        _commit(tree, None, [SYS, U1, {"role": "assistant", "content": "other-a1"}])
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2])
        assert ap.node is n1

    def test_empty_request_is_new_root(self, linear_tree):
        # [FUNCTION DOC] An empty message list gives no attach node and zero overlap.
        tree, _, _, _ = linear_tree
        ap = tree.find_attach_point([])
        assert ap.node is None
        assert ap.best_overlap == 0

    def test_search_never_mutates(self, linear_tree):
        # [FUNCTION DOC] Searching leaves each node's sequence and child count unchanged.
        tree, _, _, _ = linear_tree
        before = [(n.seq, len(n.children)) for n in tree.nodes]
        tree.find_attach_point([SYS, U1, A1, T1, A2])
        assert [(n.seq, len(n.children)) for n in tree.nodes] == before

    def test_search_handles_depth_beyond_recursion_limit(self):
        # [FUNCTION DOC] Builds a chain of the maximum node count (one user message per node),
        # lowers the interpreter recursion limit to half that count while searching for the whole
        # chain, and restores the limit afterwards. Asserts the search (which must therefore be
        # non-recursive) attaches at the last node with all messages matched.
        tree = SessionTree()
        parent = None
        for _ in range(tree_trajectory.MAX_NODES):
            parent = _commit(tree, parent, [U1])

        original_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(tree_trajectory.MAX_NODES // 2)
            ap = tree.find_attach_point([U1] * tree_trajectory.MAX_NODES)
        finally:
            sys.setrecursionlimit(original_limit)

        assert ap.node is parent
        assert ap.matched_messages == tree_trajectory.MAX_NODES


class TestModel:
    # [CLASS DOC] Tests of the tree's node data model and helpers.
    def test_case10_concurrent_commits_become_siblings(self, linear_tree):
        # [FUNCTION DOC] Two nodes created under the same parent are both its children and each
        # has that parent.
        tree, n0, _, _ = linear_tree
        s1 = _commit(tree, n0, [{"role": "user", "content": "sub A"}, A2])
        s2 = _commit(tree, n0, [{"role": "user", "content": "sub B"}, A3])
        assert s1 in n0.children and s2 in n0.children
        assert s1.parent is n0 and s2.parent is n0

    def test_full_snapshot_and_path_helpers(self, linear_tree):
        # [FUNCTION DOC] A node's token snapshot starts with its parent's snapshot; the path nodes
        # of the last node are the three chain nodes and its path messages are the seven chain
        # messages in order.
        _, n0, n1, n2 = linear_tree
        assert n2.token_ids[: len(n1.token_ids)] == n1.token_ids  # snapshot inherits prefix
        assert n2.path_nodes() == [n0, n1, n2]
        assert n2.path_messages() == [SYS, U1, A1, T1, A2, T2, A3]

    def test_token_snapshot_copies_input(self, linear_tree):
        # [FUNCTION DOC] Mutating the caller's token list after node creation does not change the
        # node's stored token ids (the node keeps a copy).
        tree, n0, _, _ = linear_tree
        token_ids = n0.token_ids + [999]
        node = tree.create_node(
            n0,
            delta_messages=[T1, A2],
            token_ids=token_ids,
            completion_span=(len(n0.token_ids), len(token_ids)),
            committed_at=3.0,
            response_id="resp-copy",
            record=n0.record,
            finish_reason="stop",
        )

        token_ids.append(1000)
        assert node.token_ids == n0.token_ids + [999]

    def test_case7_truncated_is_derived_from_finish_reason(self):
        # [FUNCTION DOC] A node finished by length is truncated; a child finished by stop is not.
        tree = SessionTree()
        node = _commit(tree, None, [U1, A1], finish_reason="length")
        assert node.truncated
        assert not _commit(tree, node, [T1, A2]).truncated

    def test_leaves_in_creation_order(self, linear_tree):
        # [FUNCTION DOC] After adding a side branch under the root node, the leaves are the chain's
        # last node followed by the side node (creation order).
        tree, n0, _, n2 = linear_tree
        side = _commit(tree, n0, [T1, {"role": "assistant", "content": "retry"}])
        assert tree.leaves() == [n2, side]

    def test_case12_node_cap_fails_loud(self, monkeypatch):
        # [FUNCTION DOC] With the module's node cap patched to two (reverted by monkeypatch),
        # creating a third node raises a value error saying the node cap was reached.
        monkeypatch.setattr(tree_trajectory, "MAX_NODES", 2)
        tree = SessionTree()
        n0 = _commit(tree, None, [U1, A1])
        _commit(tree, n0, [T1, A2])
        with pytest.raises(ValueError, match="node cap reached"):
            _commit(tree, n0, [T1, A3])

    def test_seq_is_creation_order_not_wall_clock(self):
        # [FUNCTION DOC] Two siblings committed with decreasing wall-clock times still get
        # increasing sequence numbers, and a request extending them attaches at the later-created
        # one.
        """Wall clock is decoration: ordering must survive a clock step backwards."""
        tree = SessionTree()
        n0 = _commit(tree, None, [SYS, U1, A1], committed_at=100.0)
        early_clock = _commit(tree, n0, [T1, A2], committed_at=50.0)
        later = _commit(tree, n0, [T1, A2], committed_at=99.0)
        assert early_clock.seq < later.seq
        ap = tree.find_attach_point([SYS, U1, A1, T1, A2, T2])
        assert ap.node is later
