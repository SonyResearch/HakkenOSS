from typing import TYPE_CHECKING

import numpy as np

from complex_query.core.contracts.link_predictor import LinkPredictor
from complex_query.core.entities.config.link_predictor import RandomLinkPredictorConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from query_common.entities.kg.triple import Triple


class RandomLinkPredictor(LinkPredictor[RandomLinkPredictorConfig]):
    def __init__(self, config: RandomLinkPredictorConfig):
        super().__init__(config)

        self.rng = np.random.default_rng(seed=self.config.seed)

    def predict(self, triple: "Triple") -> float:  # noqa: ARG002
        return self.rng.random()

    def predict_batch(self, triples: "Sequence[Triple]") -> "NDArray[np.float64]":
        return self.rng.random(len(triples))
