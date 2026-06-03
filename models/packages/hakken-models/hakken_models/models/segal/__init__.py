"""SeGAL — Semantic Graph-Aware Link scorer (v2, GNN-based context)."""

from hakken_models.losses.ranking_relation import RankingRelationLoss

from .base import SeGAL
from .config import ScoringConfig, SeGALConfig, TemporalEncoderConfig
from .data_module import SeGALDataModule
from .inference import SeGALInferenceWrapper
from .lightning import LitSeGAL, create_lit_segal
from .loader import SeGALArtifacts, SeGALLoader
from .temporal import TemporalEncoder

__all__ = [
    "LitSeGAL",
    "RankingRelationLoss",
    "ScoringConfig",
    "SeGAL",
    "SeGALArtifacts",
    "SeGALConfig",
    "SeGALDataModule",
    "SeGALInferenceWrapper",
    "SeGALLoader",
    "TemporalEncoder",
    "TemporalEncoderConfig",
    "create_lit_segal",
]
