# [MODULE DOC]
# Picker hook that removes abandoned (superseded) leaf samples from a session tree, plus a helper
# that compares prompt tokens of two tree nodes. Uses the tree tables in the session metadata
# (nodes with id, parent, commit time, token count and completion span; leaves with path node ids)
# and each sample's leaf descriptor and accumulated token ids in its metadata. Ids increase in
# commit order, so a larger id means a later commit.
"""Pick hook: trim rolled-back leaves."""

import logging

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def drop_rolled_back_leaves(
    leaf_samples: list[Sample], session_metadata: dict, *, sibling_prompt_must_match: bool = False
) -> list[Sample]:
    # [FUNCTION DOC]
    # Return the leaf samples that were not superseded, sorted. Pure selection (samples are not
    # modified); inputs may be reordered in the output.
    # Steps: index nodes by id and group node ids by parent (roots grouped under a null parent).
    # For each sample take its leaf descriptor (node id, parent). The "later" siblings are the
    # children of the same parent having a larger id. When the parent is null (root level) or
    # the must-match flag is set, keep only later siblings whose prompt tokens equal this leaf's
    # prompt tokens (via the prompt comparison helper). No later sibling: keep the sample.
    # Otherwise drop it, after emitting diagnostics: a warning if any later sibling has an earlier
    # commit timestamp than this leaf (clock regression; ordering still follows ids), a warning if
    # this leaf's token count exceeds the largest token count among leaves whose path includes a
    # later sibling (the maximum over an empty set would raise a value error), and an info log.
    # Result order: descending by path length (number of nodes on the leaf's path), ties broken by
    # descending node id.
    """Drop the dead leaves that rollbacks leave behind.

    When a generation goes bad, the agent re-sends the message, possibly
    edited, and the conversation continues on the new branch — but the
    abandoned attempt is still a leaf in the tree, and it must not become a
    training sample.

    The rule, leaf by leaf: a sibling branched off the same parent after it
    -> this leaf is the abandoned attempt, superseded by the re-send: trim it.
    With ``sibling_prompt_must_match``, only a sibling with the same prompt
    tokens supersedes it. Roots are siblings when their prompt tokens match:
    a re-sent first turn opens a new root. No later sibling -> keep, so roots
    with different prompts (subagents) all survive. Survivors are ordered by
    checkpoint count, then commit order, both descending.

    Example — turn 2 hit the length cap and the agent re-sent it (``seq`` is
    commit order):

        seq=0  turn 1: user asks, assistant answers
        ├── seq=1  turn 2 attempt: cut off at the length cap  (leaf) -> trim
        └── seq=2  turn 2 retry: the same user message re-sent
            └── seq=3  turn 3: the retry path continues       (leaf) -> keep
    """
    nodes = {n["id"]: n for n in session_metadata["tree"]["nodes"]}
    children: dict[int | None, list[int]] = {}  # roots are the children of None
    for n in session_metadata["tree"]["nodes"]:
        children.setdefault(n["parent"], []).append(n["id"])
    leaf_rows = session_metadata["tree"]["leaves"]

    kept: list[Sample] = []
    for sample in leaf_samples:
        descriptor = sample.metadata["leaf"]
        leaf_id, parent = descriptor["node_id"], descriptor["parent"]
        later = [sibling for sibling in children[parent] if sibling > leaf_id]
        if parent is None or sibling_prompt_must_match:
            later = [sibling for sibling in later if _same_prompt(sibling, sample, leaf_samples, nodes)]
        if not later:
            kept.append(sample)
            continue
        # Wall-clock regressions are diagnostic; `seq` is the ordering contract.
        clock_regressions = [
            (sibling, nodes[sibling]["committed_at"])
            for sibling in later
            if nodes[sibling]["committed_at"] < nodes[leaf_id]["committed_at"]
        ]
        if clock_regressions:
            logger.warning(
                "Picker detected wall-clock rollback for superseded leaf "
                "(response_id=%r, seq=%d, committed_at=%s); "
                "later siblings=%s; continuing by seq",
                descriptor["response_id"],
                leaf_id,
                nodes[leaf_id]["committed_at"],
                clock_regressions,
            )

        # Length is diagnostic only; temporal supersession is decided by `seq`.
        survivors_max = max(
            nodes[row["node_id"]]["num_tokens"]
            for row in leaf_rows
            if any(sibling in row["path_node_ids"] for sibling in later)
        )
        if nodes[leaf_id]["num_tokens"] > survivors_max:
            logger.warning(
                "Picker trimming superseded leaf (response_id=%r, seq=%d) "
                "even though it is longer than every later sibling's deepest leaf "
                "(%d > %d tokens); continuing by seq; use a custom picker to keep it",
                descriptor["response_id"],
                leaf_id,
                nodes[leaf_id]["num_tokens"],
                survivors_max,
            )
        logger.info("Picker trimmed superseded leaf seq=%d", leaf_id)
    return sorted(
        kept,
        key=lambda sample: (
            len(sample.metadata["leaf"]["path_node_ids"]),
            sample.metadata["leaf"]["node_id"],
        ),
        reverse=True,
    )


def _same_prompt(node_id: int, sample: Sample, leaf_samples: list[Sample], nodes: dict[int, dict]) -> bool:
    # [FUNCTION DOC] HELPER: True if the given node and the sample's leaf node have identical
    # prompts. The prompt length of a node is the start of its completion span. Different prompt
    # lengths give false. Otherwise find the first leaf sample whose path contains the node (none
    # gives false) and compare the first prompt-length accumulated token ids of that sample and of
    # the given sample.
    """Whether ``node_id``'s prompt tokens equal those of ``sample``'s leaf.

    Tokens come from any leaf sample whose path covers the node; a node that no
    leaf sample covers, e.g. one whose leaves all truncated away, matches nothing.
    """
    prompt_len = nodes[sample.metadata["leaf"]["node_id"]]["completion_span"][0]
    if nodes[node_id]["completion_span"][0] != prompt_len:
        return False
    covering = next((s for s in leaf_samples if node_id in s.metadata["leaf"]["path_node_ids"]), None)
    return (
        covering is not None
        and covering.metadata["accumulated_token_ids"][:prompt_len]
        == sample.metadata["accumulated_token_ids"][:prompt_len]
    )
