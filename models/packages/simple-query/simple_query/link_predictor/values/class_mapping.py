from typing import TYPE_CHECKING

from simple_query.link_predictor.entities.configs import (
    ApiBasedLinkPredictorConfig,
    RandomLinkPredictorConfig,
)
from simple_query.link_predictor.impl.api_based import ApiBasedLinkPredictor
from simple_query.link_predictor.impl.random import RandomLinkPredictor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simple_query.link_predictor.base import LinkPredictor
    from simple_query.link_predictor.entities.configs import LinkPredictorConfigBase

CLASS_MAPPING: "Mapping[type[LinkPredictorConfigBase], type[LinkPredictor]]" = {
    ApiBasedLinkPredictorConfig: ApiBasedLinkPredictor,
    RandomLinkPredictorConfig: RandomLinkPredictor,
}
