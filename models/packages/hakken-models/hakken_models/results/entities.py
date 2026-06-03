from __future__ import annotations

from dataclasses import dataclass

OriginalName = str
ColumnName = str


@dataclass(frozen=True, slots=True)
class ModelInfo:
    display_name: str
    position: int
    color: str  # matplotlib color (hex, name, etc.)


@dataclass
class SaveFigureConfig:
    path: str
    dpi: int = 300
    bbox_inches: str = "tight"


@dataclass
class LegendConfig:
    loc: str = "upper right"
    bbox_to_anchor: tuple[float, float] = (1.25, 1.15)
    frameon: bool = False
    ncol: int | None = None


@dataclass(frozen=True, slots=True)
class MetricInfo:
    display_name: str
    higher_is_better: bool
