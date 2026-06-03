from hakken_models.registries.base import Registry

from .base import ScoreFn
from .complex import ComplExScore
from .conv_kb import ConvKBScore
from .distmult import DistMultScore
from .rotate import RotatEScore


class ScoreFnRegistry(Registry[ScoreFn]):
    pass


score_fn_registry = ScoreFnRegistry("ScoreFn")


score_fn_registry.register_class(ComplExScore)
score_fn_registry.register_class(DistMultScore)
score_fn_registry.register_class(RotatEScore)
score_fn_registry.register_class(ConvKBScore)


__all__ = ["ComplExScore", "ConvKBScore", "DistMultScore", "RotatEScore", "score_fn_registry"]
