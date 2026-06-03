from typing import TYPE_CHECKING

import numpy as np

from simple_query.link_predictor.base import LinkPredictor
from simple_query.link_predictor.entities.configs import (
    RandomLinkPredictorConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from query_common.entities.kg.triple import Triple


class RandomLinkPredictor(LinkPredictor[RandomLinkPredictorConfig]):
    def __init__(self, config: RandomLinkPredictorConfig) -> None:
        super().__init__(config)

        self.rng = np.random.default_rng(seed=self.config.seed)

    def _predict(self, triples: "Sequence[Triple]") -> list[float]:
        return self.rng.random(len(triples)).astype(float).tolist()  # type: ignore
