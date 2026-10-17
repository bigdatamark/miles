# [PACKAGE DOC]
# Score-centering policy-gradient loss package. Re-exports the loss function with its input and
# metric dataclasses (estimator), the importance weight function (importance_sampling) and the
# selected-token log-prob helpers (selected_log_probs). Internal mask helpers live in masks.
"""Score centering from arXiv:2609.20807, Appendix A."""

from miles.backends.training_utils.loss.hub.score_centering.estimator import (
    ScoreCenteringInputs,
    ScoreCenteringMetrics,
    score_centering_loss,
)
from miles.backends.training_utils.loss.hub.score_centering.importance_sampling import importance_weights
from miles.backends.training_utils.loss.hub.score_centering.selected_log_probs import (
    selected_log_probs,
    selected_log_probs_and_entropy,
)

__all__ = [
    "ScoreCenteringInputs",
    "ScoreCenteringMetrics",
    "importance_weights",
    "score_centering_loss",
    "selected_log_probs",
    "selected_log_probs_and_entropy",
]
