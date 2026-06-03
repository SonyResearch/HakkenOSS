from pydantic import BaseModel, Field
from strenum import StrEnum


class NegativeAggregation(StrEnum):
    HARDEST = "hardest"  # Use max / worst negative (contrastive focus)
    MEAN = "mean"  # Average all negatives (smoother gradient)


class NegativeStrategyConfig(BaseModel):
    """
    KGE: single source of truth for hardest vs mean aggregation over sampled
    negatives (injected into ``RankingRelationLoss`` kwargs as ``neg_strategy``).

    This does not replace ``negative_sampler`` (how corruptions are drawn).
    """

    name: NegativeAggregation = Field(
        default=NegativeAggregation.HARDEST,
        description=(
            "How to aggregate multiple negative losses:\n"
            "  - 'hardest': take the maximum (worst) score\n"
            "  - 'mean':    average over all negatives"
        ),
    )
    kwargs: dict = Field(
        default_factory=dict,
        description="Additional parameters for the negative strategy (if needed)",
    )

    @property
    def uses_hard_negatives(self) -> bool:
        return self.name == NegativeAggregation.HARDEST
