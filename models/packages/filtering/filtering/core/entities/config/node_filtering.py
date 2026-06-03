from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from filtering.core.entities.kg import EdgeDirection, YearRange
from filtering.core.values.types import NodeFilteringType

if TYPE_CHECKING:
    from typing import Self


class NodeFilteringConfigBase(BaseModel):
    config_type: NodeFilteringType


class EntropyNodeFilteringConfig(NodeFilteringConfigBase):
    config_type: Literal[NodeFilteringType.ENTROPY] = NodeFilteringType.ENTROPY

    degree_direction: EdgeDirection = EdgeDirection.ALL
    """Direction of degrees to be considered."""
    year_range: YearRange = YearRange(2010, 2024)
    """Range of year to consider (left-inclusive)."""
    year_window_size: int = 1
    """Size of year window. It will use edges made between
    (year_range.start, year_range.end + year_window_size) in computing the initial degree,
    and in turn calculating the slope.
    """

    @model_validator(mode="after")
    def check_year_range_values(self) -> "Self":
        if self.year_range.start + self.year_window_size >= self.year_range.end:
            raise ValueError(
                "`year_range.start` should be larger than `year_range.end + year_window_size`."
            )
        return self


class RecencyNodeFilteringConfig(NodeFilteringConfigBase):
    config_type: Literal[NodeFilteringType.RECENCY] = NodeFilteringType.RECENCY

    degree_direction: EdgeDirection = EdgeDirection.ALL
    """Direction of degrees to be considered."""
    year_range: YearRange = YearRange(2010, 2024)
    """Range of year to consider (left-inclusive)."""
    year_window_size: int = 1
    """Size of year window. It will split the degrees in chunks using this size
    and compute recency scores based on those chunks.
    `year_range[1] - year_range[0]` should be divisible by this value."""
    recency_min_weight: float = 0.1
    """Minimum weight to be used in recency filtering."""

    @model_validator(mode="after")
    def check_year_range_values(self) -> "Self":
        if self.year_range.start + self.year_window_size >= self.year_range.end:
            raise ValueError(
                "`year_range.end` should be larger than `year_range.start + year_window_size`."
            )
        return self


class RandomNodeFilteringConfig(NodeFilteringConfigBase):
    config_type: Literal[NodeFilteringType.RANDOM] = NodeFilteringType.RANDOM

    random_seed: int | None = None


NodeFilteringConfig = Annotated[
    EntropyNodeFilteringConfig | RecencyNodeFilteringConfig | RandomNodeFilteringConfig,
    Field(discriminator="config_type"),
]
