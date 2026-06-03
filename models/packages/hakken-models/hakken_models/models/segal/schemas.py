"""SeGAL output schemas."""

from typing import NamedTuple

from torch import Tensor


class ScoreBatchOutput(NamedTuple):
    """Output of :meth:`SeGAL.score_batch`."""

    pos_scores: Tensor
    neg_scores: Tensor


class ScoreStepOutput(NamedTuple):
    """Container for the outputs of :meth:`LitSeGAL._score_step`."""

    pos_scores: Tensor
    neg_scores: Tensor
    rel_logits: Tensor | None = None
    rel_labels: Tensor | None = None
