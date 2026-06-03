from hakken_explainer.constants import RerankStrategy, ScoreType
from hakken_explainer.entities.config import ScoreTypeConfig
from pydantic import BaseModel, Field


class PathExplainerRequest(BaseModel):
    triples_to_probe: list[list[str]] = Field(
        ...,
        description="List of triples to explain, where each triple is [subject, relation, object]",
        min_length=1,
    )
    num_explanations: int = Field(
        default=10,
        description="Maximum number of explanations to generate per triple",
        ge=1,
    )
    explanation_configs: list[ScoreTypeConfig] = Field(
        default_factory=lambda: [ScoreTypeConfig(type=ScoreType.SUFFICIENT, batch_size=32)],
        description="Explanation configurations specifying types and batch sizes",
    )

    min_explanation_length: int | None = Field(
        default=None,
        description=(
            "Minimum explanation length required to process a triple. "
            "Triples with explanation length below this threshold will be skipped "
            "(None for no filtering)"
        ),
        ge=1,
    )

    allowed_relation_ids: list[str] | None = Field(
        default=None,
        description="List of relations to include in explanations (None for all relations)",
    )
    rerank_strategy: RerankStrategy = Field(
        default=RerankStrategy.UNIQUE_PATHWAYS,
        description="Strategy for ranking final results (e.g., 'scores' or 'unique_pathways')",
    )


class PathExplainerLengthRequest(BaseModel):
    triples_to_probe: list[list[str]]
