# [PACKAGE DOC]
# Built-in picker hooks for the v2 session server's sample collection. A picker receives the
# list of per-leaf raw samples and the session metadata and returns a subset (selection only,
# order may change, samples must not be rewritten). The package re-exports two pickers: one that
# drops leaves superseded by a later sibling branch (rollbacks) and one that does so only when
# the sibling re-sent the same prompt tokens (retries; the default). Selected by an import path in
# the launch options.
"""Pick hooks for the session samples op — this is the customization point.

Point ``--session-sample-picker-path`` at any ``fn(leaf_samples,
session_metadata) -> list[Sample]``: a pure selection over the per-leaf raw
samples (drop or reorder, never rewrite). Public inputs are each sample's
``metadata["leaf"]`` descriptor and ``metadata["accumulated_token_ids"]``, and
the structural tree tables in ``session_metadata``. Runs synchronously inside
the session server process.
"""

from miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves import drop_rolled_back_leaves
from miles.rollout.session.v2.picker_hub.drop_same_prompt_retries import drop_same_prompt_retries

__all__ = ["drop_rolled_back_leaves", "drop_same_prompt_retries"]
