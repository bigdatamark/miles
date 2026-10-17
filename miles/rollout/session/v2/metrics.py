# [MODULE DOC]
# Per-session rollout metrics for the v2 session server. Defines the metadata key under which
# the metrics are published and a builder that aggregates speculative decoding statistics over
# all tree nodes of a session.
from __future__ import annotations

from typing import TYPE_CHECKING

from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.session.v2.tree_trajectory import TrajectoryNode

SESSION_ROLLOUT_METRICS_KEY = "session_rollout_metrics"


def build_session_rollout_metrics(session_id: str, nodes: list[TrajectoryNode]) -> dict:
    # [FUNCTION DOC] Accumulate speculative-decoding info over every node's first-choice meta
    # info (in order) into one fresh accumulator and return a mapping with the session id and a
    # metrics mapping holding the accumulator as a dictionary under the speculative info key.
    # A node whose response lacks the first choice or meta info raises a lookup error.
    spec_info = Sample.SpecInfo()
    for node in nodes:
        spec_info.add(node.record.response["choices"][0]["meta_info"])
    return {
        "session_id": session_id,
        "metrics": {"spec_info": spec_info.to_dict()},
    }
