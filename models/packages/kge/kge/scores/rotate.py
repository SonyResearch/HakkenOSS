from __future__ import annotations

from typing import cast

import torch

from kge.common.types import FloatTensor2D

from .base import ScoreFn


class RotatEScore(ScoreFn):
    """RotatE scoring function for knowledge graph embeddings.

    RotatE models each relation as a rotation from the source entity to the target
    entity in complex vector space. The key insight is that relations are represented
    as rotations with unit modulus, allowing the model to handle various relation
    patterns including symmetry, antisymmetry, inversion, and composition.

    The scoring function is defined as:
    score(h, r, t) = -||h ∘ r - t||

    where:
    - h, r, t are complex embeddings
    - ∘ denotes element-wise (Hadamard) product
    - |r_i| = 1 for all relation embedding components (unit modulus constraint)

    Key properties:
    - Symmetry: r_symmetric has phase 0 or π
    - Antisymmetry: general rotation angles
    - Inversion: r_inverse = conjugate(r) or r + r_inverse = 0 (in phase)
    - Composition: r_3 = r_1 + r_2 (in phase space)

    References:
        Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation
        in Complex Space." ICLR 2019. https://arxiv.org/pdf/1902.10197
    """

    def __init__(self, epsilon: float = 2.0) -> None:
        """Initialize the RotatE scoring function.

        Args:
            epsilon: Small value to avoid division by zero.
        """
        super().__init__()
        self.epsilon = epsilon

    def get_complex_dim(self, embeddings: FloatTensor2D) -> int:
        return embeddings.shape[1] // 2

    def _to_complex(self, embeddings: FloatTensor2D) -> torch.Tensor:
        """Convert real embeddings to complex format.

        Args:
            embeddings: Real embeddings with shape (..., embedding_dim).

        Returns:
            Complex embeddings with shape (..., complex_dim).
        """
        complex_dim = self.get_complex_dim(embeddings)
        real = embeddings[..., :complex_dim]
        imag = embeddings[..., complex_dim:]
        return torch.complex(real, imag)

    def _normalize_relation_embeddings(self, r_emb: FloatTensor2D) -> FloatTensor2D:
        """Normalize relation embeddings to have unit modulus.

        Relations in RotatE are constrained to have unit modulus, representing
        pure rotations without scaling.

        Args:
            r_emb: Relation embeddings with shape (..., embedding_dim).

        Returns:
            Normalized relation embeddings with unit modulus.
        """
        r_complex = self._to_complex(r_emb)

        # Normalize to unit modulus (|r_i| = 1)
        r_normalized = r_complex / (torch.abs(r_complex) + self.epsilon)

        real_part = r_normalized.real
        imag_part = r_normalized.imag

        return torch.cat([real_part, imag_part], dim=-1)

    def _rotate_score(
        self, h_emb: FloatTensor2D, r_emb: FloatTensor2D, t_emb: FloatTensor2D
    ) -> FloatTensor2D:
        """Compute RotatE score: -||h ∘ r - t||.

        Args:
            h_emb: Head entity embeddings.
            r_emb: Relation embeddings (will be normalized).
            t_emb: Tail entity embeddings.

        Returns:
            Negative distance scores (higher is better).
        """
        r_normalized = self._normalize_relation_embeddings(r_emb)

        h_complex = self._to_complex(h_emb)
        r_complex = self._to_complex(r_normalized)
        t_complex = self._to_complex(t_emb)

        # Compute rotation: h ∘ r
        rotated = h_complex * r_complex

        # Compute distance: ||h ∘ r - t||
        diff = rotated - t_complex
        distance = torch.norm(diff, p=2, dim=-1)

        # Return negative distance (higher score = better)
        return cast("FloatTensor2D", -distance)

    def subjects(
        self,
        r_emb: FloatTensor2D,
        o_emb: FloatTensor2D,
        entity_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        """Compute RotatE scores for all possible subjects.

        For RotatE, we need to solve: h ∘ r = t
        Therefore: h = t ∘ r^(-1) = t ∘ conjugate(r)
        Score = -||h ∘ r - t|| = -||entity ∘ r - t||
        """

        embedding_dim = entity_embeddings.shape[1]
        batch_size = r_emb.shape[0]
        num_entities = entity_embeddings.shape[0]

        r_expanded = r_emb.unsqueeze(1).expand(batch_size, num_entities, -1)
        o_expanded = o_emb.unsqueeze(1).expand(batch_size, num_entities, -1)
        entity_expanded = entity_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        r_flat = r_expanded.reshape(-1, embedding_dim)
        o_flat = o_expanded.reshape(-1, embedding_dim)
        entity_flat = entity_expanded.reshape(-1, embedding_dim)

        scores_flat = self._rotate_score(entity_flat, r_flat, o_flat)

        return scores_flat.reshape(batch_size, num_entities)

    def relations(
        self,
        s_emb: FloatTensor2D,
        o_emb: FloatTensor2D,
        relation_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        """Compute RotatE scores for all possible relations.

        For RotatE: score = -||s ∘ r - o||
        """

        batch_size = s_emb.shape[0]
        embedding_dim = s_emb.shape[1]
        num_relations = relation_embeddings.shape[0]

        s_expanded = s_emb.unsqueeze(1).expand(batch_size, num_relations, -1)
        o_expanded = o_emb.unsqueeze(1).expand(batch_size, num_relations, -1)
        relation_expanded = relation_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        s_flat = s_expanded.reshape(-1, embedding_dim)
        o_flat = o_expanded.reshape(-1, embedding_dim)
        relation_flat = relation_expanded.reshape(-1, embedding_dim)

        scores_flat = self._rotate_score(s_flat, relation_flat, o_flat)

        return scores_flat.reshape(batch_size, num_relations)

    def objects(
        self,
        s_emb: FloatTensor2D,
        r_emb: FloatTensor2D,
        entity_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        """Compute RotatE scores for all possible objects.

        For RotatE: score = -||s ∘ r - entity||
        """
        batch_size = s_emb.shape[0]
        embedding_dim = s_emb.shape[1]
        num_entities = entity_embeddings.shape[0]

        s_expanded = s_emb.unsqueeze(1).expand(batch_size, num_entities, -1)
        r_expanded = r_emb.unsqueeze(1).expand(batch_size, num_entities, -1)
        entity_expanded = entity_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        s_flat = s_expanded.reshape(-1, embedding_dim)
        r_flat = r_expanded.reshape(-1, embedding_dim)
        entity_flat = entity_expanded.reshape(-1, embedding_dim)

        scores_flat = self._rotate_score(s_flat, r_flat, entity_flat)

        return scores_flat.reshape(batch_size, num_entities)

    def all(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        """Compute RotatE scores for specific subject-relation-object triples.

        Args:
            s_emb: Subject embeddings with shape (batch_size, embedding_dim).
            r_emb: Relation embeddings with shape (batch_size, embedding_dim).
            o_emb: Object embeddings with shape (batch_size, embedding_dim).

        Returns:
            Scores with shape (batch_size,).
        """
        return self._rotate_score(s_emb, r_emb, o_emb).unsqueeze(-1)
