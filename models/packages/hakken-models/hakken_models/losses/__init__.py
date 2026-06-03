from torch.nn import BCEWithLogitsLoss, MarginRankingLoss

from ._registry import loss_fn_registry
from .aggregated_ranking import AggregatedRankingLoss
from .asymmetric_loss import AsymmetricLoss
from .focal_loss import FLWithLogitsLoss
from .nssa import NSSALoss
from .ranking_relation import RankingRelationLoss
from .relation_weights import compute_pos_weight_from_relation_labels

loss_fn_registry.register_class(AsymmetricLoss)
loss_fn_registry.register_class(FLWithLogitsLoss)
loss_fn_registry.register_class(BCEWithLogitsLoss)
loss_fn_registry.register_class(NSSALoss)
loss_fn_registry.register_class(MarginRankingLoss)
loss_fn_registry.register_class(AggregatedRankingLoss)
loss_fn_registry.register_class(RankingRelationLoss)

__all__ = [
    "AggregatedRankingLoss",
    "AsymmetricLoss",
    "FLWithLogitsLoss",
    "NSSALoss",
    "RankingRelationLoss",
    "compute_pos_weight_from_relation_labels",
    "loss_fn_registry",
]
