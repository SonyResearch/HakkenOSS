from typing import Protocol

from kge.models.gnn import GNNKGE
from torch import Tensor

from hakken_explainer.constants import ScoreType

from .base import ExplainerScore
from .necessary import NecessaryScore
from .sufficient import SufficientScore


class ExplinerScoreBuilder(Protocol):
    @staticmethod
    def from_kwargs(score_type: ScoreType, context_kg: Tensor, model: GNNKGE) -> ExplainerScore:
        """Create an explanation scorer based on the specified type.

        Args:
            score_type: The type of explanation scoring to use
            context_kg: Knowledge graph tensor for context [num_facts, 3]
            model: Trained GNN model for scoring predictions

        Returns:
            ExplainerScore: Configured scorer instance

        Raises:
            ValueError: If score_type is not supported
        """
        if score_type == ScoreType.NECESSARY:
            return NecessaryScore(context_kg, model)
        if score_type == ScoreType.SUFFICIENT:
            return SufficientScore(context_kg, model)
        msg = f"Unsupported explanation type: {score_type}"
        raise ValueError(msg)
