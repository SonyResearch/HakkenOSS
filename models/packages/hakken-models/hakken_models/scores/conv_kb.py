from __future__ import annotations

import torch
import torch.nn.functional as torch_f
from torch import nn

from .base import ScoreFn


class ConvKBScore(ScoreFn):
    """Scoring function implementing ConvKB (Nguyen et al., NAACL 2018).

    Uses 1 x 3 convolutions across the subject-relation-object triple.
    """

    def __init__(
        self,
        emb_dim: int,
        num_filters: int = 50,
        kernel_size: tuple[int, int] = (1, 3),
    ) -> None:
        super().__init__()
        self.emb_dim: int = emb_dim
        self.num_filters: int = num_filters

        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,  # (1, 3) → conv across the triple axis
        )
        self.fc: nn.Linear = nn.Linear(num_filters * emb_dim, 1)

    def _score_triples(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
        o: torch.Tensor,
    ) -> torch.Tensor:
        """Core scoring logic for a batch of (s, r, o) triples.

        Args:
            s: Subject embeddings (batch_size, emb_dim)
            r: Relation embeddings (batch_size, emb_dim)
            o: Object embeddings (batch_size, emb_dim)

        Returns:
            torch.Tensor: Scores of shape (batch_size, 1)
        """
        batch_size: int = s.shape[0]
        if batch_size == 0:
            return torch.empty(0, 1, device=s.device, dtype=s.dtype)

        # Stack along the triple dimension → (batch_size, emb_dim, 3)
        x: torch.Tensor = torch.stack([s, r, o], dim=2)

        # Add channel dimension → (batch_size, 1, emb_dim, 3)
        x = x.unsqueeze(1)

        # Convolution → (batch_size, num_filters, emb_dim, 1)
        x = self.conv(x)
        x = torch_f.relu(x)

        # Flatten → (batch_size, num_filters * emb_dim)
        x = x.view(batch_size, -1)

        # Final projection → (batch_size, 1)
        return self.fc(x)

    def all(
        self,
        s_emb: torch.Tensor,
        r_emb: torch.Tensor,
        o_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score complete (s, r, o) triples.

        Args:
            s_emb: (batch_size, emb_dim)
            r_emb: (batch_size, emb_dim)
            o_emb: (batch_size, emb_dim)

        Returns:
            torch.Tensor: Scores of shape (batch_size, 1)
        """
        return self._score_triples(s_emb, r_emb, o_emb)

    def subjects(
        self,
        r_emb: torch.Tensor,
        o_emb: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score all possible subjects: (?, r, o)

        Args:
            r_emb: (batch_size, emb_dim)
            o_emb: (batch_size, emb_dim)
            entity_embeddings: (num_entities, emb_dim)

        Returns:
            torch.Tensor: Scores of shape (batch_size, num_entities)
        """
        batch_size: int = r_emb.shape[0]
        num_entities: int = entity_embeddings.shape[0]

        # Broadcast to (batch_size, num_entities, emb_dim)
        s = entity_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        r = r_emb.unsqueeze(1).expand(-1, num_entities, -1)
        o = o_emb.unsqueeze(1).expand(-1, num_entities, -1)

        # Flatten batch and candidates → (batch_size * num_entities, emb_dim)
        scores: torch.Tensor = self._score_triples(
            s.reshape(-1, self.emb_dim),
            r.reshape(-1, self.emb_dim),
            o.reshape(-1, self.emb_dim),
        )

        return scores.view(batch_size, num_entities)

    def objects(
        self,
        s_emb: torch.Tensor,
        r_emb: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score all possible objects: (s, r, ?)

        Args:
            s_emb: (batch_size, emb_dim)
            r_emb: (batch_size, emb_dim)
            entity_embeddings: (num_entities, emb_dim)

        Returns:
            torch.Tensor: Scores of shape (batch_size, num_entities)
        """
        batch_size: int = s_emb.shape[0]
        num_entities: int = entity_embeddings.shape[0]

        s = s_emb.unsqueeze(1).expand(-1, num_entities, -1)
        r = r_emb.unsqueeze(1).expand(-1, num_entities, -1)
        o = entity_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        scores: torch.Tensor = self._score_triples(
            s.reshape(-1, self.emb_dim),
            r.reshape(-1, self.emb_dim),
            o.reshape(-1, self.emb_dim),
        )

        return scores.view(batch_size, num_entities)

    def relations(
        self,
        s_emb: torch.Tensor,
        o_emb: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score all possible relations: (s, ?, o)

        Args:
            s_emb: (batch_size, emb_dim)
            o_emb: (batch_size, emb_dim)
            relation_embeddings: (num_relations, emb_dim)

        Returns:
            torch.Tensor: Scores of shape (batch_size, num_relations)
        """
        batch_size: int = s_emb.shape[0]
        num_relations: int = relation_embeddings.shape[0]

        s = s_emb.unsqueeze(1).expand(-1, num_relations, -1)
        r = relation_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        o = o_emb.unsqueeze(1).expand(-1, num_relations, -1)

        scores: torch.Tensor = self._score_triples(
            s.reshape(-1, self.emb_dim),
            r.reshape(-1, self.emb_dim),
            o.reshape(-1, self.emb_dim),
        )

        return scores.view(batch_size, num_relations)
