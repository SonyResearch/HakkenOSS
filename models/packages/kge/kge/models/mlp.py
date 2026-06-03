from __future__ import annotations

from gnn.mlp import MLPConfig
from hakken_ml_toolkit.ml_utils.extras import TensorCreator
from omegaconf import MISSING

from kge.common.types import FloatTensor2D
from kge.models.er_model import ERModel, ERModelConfig
from kge.scores import MLPScore


class MLPKGEConfig(ERModelConfig):
    name: str = "mlp_kge"
    mlp_config: MLPConfig = MISSING


class MLPKGE(ERModel[MLPKGEConfig]):
    def __init__(self, config: MLPKGEConfig):
        super().__init__(config)

        config.mlp_config.input_dim = 3 * config.embedding_dim

        self.score_fn = MLPScore(config.mlp_config)

    def _score_objects(self, s_emb: FloatTensor2D, r_emb: FloatTensor2D) -> FloatTensor2D:
        device = s_emb.device
        entities = TensorCreator.long_arange(self.config.num_entities, device=device)
        entity_embeddings = self.entity_embeddings(entities)

        # Calculate scores and reshape
        return self.score_fn.objects(s_emb, r_emb, entity_embeddings=entity_embeddings)

    def _score_subjects(self, r_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        device = o_emb.device
        entities = TensorCreator.long_arange(self.config.num_entities, device=device)
        entity_embeddings = self.entity_embeddings(entities)

        return self.score_fn.subjects(r_emb, o_emb, entity_embeddings=entity_embeddings)

    def _score_relations(self, s_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        device = o_emb.device
        relations = TensorCreator.long_arange(self.config.num_relations, device=device)
        relation_embeddings = self.relation_embeddings(relations)

        return self.score_fn.relations(
            s_emb=s_emb, o_emb=o_emb, relation_embeddings=relation_embeddings
        )

    def _score(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        return self.score_fn.all(s_emb, r_emb, o_emb)
