import torch
from torch import nn

from hakken_models.core.configs.model import KGEConfig
from hakken_models.scores import score_fn_registry
from hakken_models.scores.base import ScoreFn


class KGE(nn.Module):
    """
    Base class for Knowledge Graph Embedding models.

    All KGE models should inherit from this class and implement the forward method
    to compute scores for batches of (head, relation, tail) triples.
    """

    def __init__(
        self, num_entities: int, num_relations: int, embedding_dim: int, score_fn: ScoreFn
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim, padding_idx=None)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim, padding_idx=None)

        self.score_fn = score_fn

    def forward(self, facts_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for KGE models.

        Args:
            facts_batch: Tensor of shape [num_facts, >3] where:
                        - facts_batch.ndim == 2
                        - facts_batch[:, 0] contains head entity indices
                        - facts_batch[:, 1] contains relation indices
                        - facts_batch[:, 2] contains tail entity indices
                        - Additional columns (if any) are ignored

        Returns:
            Tensor of shape (num_facts,) containing scores for each triple.
            Higher scores indicate more plausible triples.
        """
        assert facts_batch.ndim == 2, f"Expected 2D tensor, got {facts_batch.ndim}D"
        assert facts_batch.shape[1] >= 3, f"Expected at least 3 columns, got {facts_batch.shape[1]}"

        head = facts_batch[:, 0].long()
        relation = facts_batch[:, 1].long()
        tail = facts_batch[:, 2].long()

        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)

        return self.score_fn.all(head_emb, relation_emb, tail_emb).squeeze(-1)

    def score(self, facts_batch: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of triples. Alias for forward method.

        Args:
            facts_batch: Tensor of shape [num_facts, >3] containing triples

        Returns:
            Tensor of shape (num_facts,) containing scores for the triples
        """
        return self.forward(facts_batch)

    def score_relations(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        Score all possible relations for given (head, tail) pairs.

        Args:
            head: Tensor of shape (batch_size,) containing head entity indices
            tail: Tensor of shape (batch_size,) containing tail entity indices

        Returns:
            Tensor of shape (batch_size, num_relations) containing scores for
            each (head, tail) pair across all possible relations.
        """
        head_emb = self.entity_embeddings(head.long())
        tail_emb = self.entity_embeddings(tail.long())

        all_relation_emb = self.relation_embeddings.weight

        return self.score_fn.relations(head_emb, tail_emb, all_relation_emb)

    def score_subjects(self, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        Score all possible head entities (subjects) for given (relation, tail) pairs.

        Args:
            relation: Tensor of shape (batch_size,) containing relation indices
            tail: Tensor of shape (batch_size,) containing tail entity indices

        Returns:
            Tensor of shape (batch_size, num_entities) containing scores for
            each (relation, tail) pair across all possible head entities.
        """
        relation_emb = self.relation_embeddings(relation.long())
        tail_emb = self.entity_embeddings(tail.long())

        all_entity_emb = self.entity_embeddings.weight

        return self.score_fn.subjects(relation_emb, tail_emb, all_entity_emb)

    def score_objects(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """
        Score all possible tail entities (objects) for given (head, relation) pairs.

        Args:
            head: Tensor of shape (batch_size,) containing head entity indices
            relation: Tensor of shape (batch_size,) containing relation indices

        Returns:
            Tensor of shape (batch_size, num_entities) containing scores for
            each (head, relation) pair across all possible tail entities.
        """

        head_emb = self.entity_embeddings(head.long())
        relation_emb = self.relation_embeddings(relation.long())

        all_entity_emb = self.entity_embeddings.weight

        return self.score_fn.objects(head_emb, relation_emb, all_entity_emb)

    @classmethod
    def from_config(cls, config: KGEConfig, num_entities: int, num_relations: int) -> "KGE":
        score_fn = score_fn_registry.create(config.score_fn_name, **config.score_fn_kwargs)
        return KGE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=config.embedding_dim,
            score_fn=score_fn,
        )
