"""Test suite for RandomKGE implementation."""

import pytest

from kge.common.types import LongTensor2D
from kge.models.base import KGEI
from kge.models.config import KGEConfig
from kge.models.random import RandomKGE
from tests.kge.base import BaseKGEITest


class TestRandomKGE(BaseKGEITest):
    """Test class for RandomKGE implementation."""

    @pytest.fixture
    def model(self):
        """Fixture providing a RandomKGE model instance."""

        config = KGEConfig(num_entities=100, num_relations=10, embedding_dim=32)
        return RandomKGE(config)

    def test_normalize_scores_without_scaler_raises(
        self, model: KGEI[KGEConfig], sro_batch: LongTensor2D
    ):
        model.eval()
        scores = model.score(sro_batch)
        model.normalize_scores(scores)
