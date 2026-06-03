from __future__ import annotations

from pydantic import BaseModel, Field

from kge.common.constants import TargetType


class NegativeSamplerConfig(BaseModel):
    num_negatives: int = Field(
        default=1,
        description="Number of negative samples to generate per positive sample",
    )

    filter_triples: list[str] | None = Field(
        default=None,
        description="""Optional list of triples to filter out during negative sampling. 
        This prevents generating negative samples that are actually positive examples.""",
    )

    corruption_scheme: list[TargetType] = Field(
        default_factory=lambda: [TargetType.OBJECT],
        description="""
        Defines which parts of the triple to corrupt during negative sampling. 
        Default is to only corrupt the OBJECT position in (subject, relation, object) 
        triples.""",
    )
