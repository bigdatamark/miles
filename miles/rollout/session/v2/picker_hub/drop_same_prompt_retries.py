# [MODULE DOC]
# The default picker hook for the v2 session server: a thin wrapper over the rolled-back-leaf
# picker with prompt matching switched on, so only re-sends of an identical prompt supersede an
# earlier attempt.
"""The default pick hook: trim retries, i.e. identical re-sends."""

from miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves import drop_rolled_back_leaves
from miles.utils.types import Sample


def drop_same_prompt_retries(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    # [FUNCTION DOC] Delegate to the rolled-back-leaf picker with the sibling-prompt-must-match
    # option on, so a later sibling only supersedes a leaf if their prompt tokens are equal at
    # every level (not just roots). Returns its result unchanged.
    """Drop the leaves that retries leave behind; a retry re-sends the same prompt tokens.

    Like ``drop_rolled_back_leaves``, but a later sibling with a different
    request is a new branch, so both leaves stay samples.
    """
    return drop_rolled_back_leaves(leaf_samples, session_metadata, sibling_prompt_must_match=True)
