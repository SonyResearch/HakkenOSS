from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from kge.models.er_model import ERModel, ERModelConfig

if TYPE_CHECKING:
    from kge.common.types import FloatTensor2D


class ComplExConfig(ERModelConfig):
    name: str = "complex"


class ComplEx(ERModel[ComplExConfig]):
    def __init__(self, config: ComplExConfig):
        super().__init__(config)

    @classmethod
    def get_config_class(cls) -> type[ComplExConfig]:
        return ComplExConfig

    def embedding_dim(self) -> int:
        return 2 * self.config.embedding_dim

    def _get_entity_embeddings(self, num_entities: int, embedding_dim: int) -> nn.Embedding:
        return nn.Embedding(num_entities, 2 * embedding_dim)

    def _get_relation_embeddings(self, num_entities: int, embedding_dim: int) -> nn.Embedding:
        return nn.Embedding(num_entities, 2 * embedding_dim)

    def _split_complex(self, x: FloatTensor2D) -> tuple[torch.Tensor, torch.Tensor]:
        all_e_re, all_e_im = torch.chunk(x, 2, dim=-1)
        return all_e_re, all_e_im

    def _score_subjects(self, r_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        r_re, r_im = self._split_complex(r_emb)
        o_re, o_im = self._split_complex(o_emb)

        ro_real = r_re * o_re + r_im * o_im
        ro_imaginary = r_re * o_im - r_im * o_re

        all_emb_real, all_emb_imaginary = self._split_complex(self._entity_embeddings.weight)

        return torch.matmul(ro_real, all_emb_real.t()) + torch.matmul(
            ro_imaginary, all_emb_imaginary.t()
        )

    def _score_relations(self, s_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        s_re, s_im = self._split_complex(s_emb)
        o_re, o_im = self._split_complex(o_emb)

        so_real = s_re * o_re + s_im * o_im
        so_imaginary = s_im * o_re - s_re * o_im

        all_rel_real, all_rel_imaginary = self._split_complex(self._relation_embeddings.weight)

        return torch.matmul(so_real, all_rel_real.t()) + torch.matmul(
            so_imaginary, all_rel_imaginary.t()
        )

    def _score_objects(self, s_emb: FloatTensor2D, r_emb: FloatTensor2D) -> FloatTensor2D:
        s_re, s_im = self._split_complex(s_emb)
        r_re, r_im = self._split_complex(r_emb)

        sr_real = s_re * r_re - s_im * r_im
        sr_imaginary = s_re * r_im + s_im * r_re

        all_emb_real, all_emb_imaginary = self._split_complex(self._entity_embeddings.weight)

        return torch.matmul(sr_real, all_emb_real.t()) + torch.matmul(
            sr_imaginary, all_emb_imaginary.t()
        )

    def _score(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        # Split into real and imaginary parts
        s_re, s_im = self._split_complex(s_emb)
        r_re, r_im = self._split_complex(r_emb)
        o_re, o_im = self._split_complex(o_emb)

        real_real_real = s_re * r_re * o_re
        im_real_im = s_im * r_re * o_im
        real_im_im = s_re * r_im * o_im
        im_im_real = s_im * r_im * o_re

        scores = real_real_real + im_real_im + real_im_im - im_im_real

        # Sum over the embedding dimension
        return torch.sum(scores, dim=-1, keepdim=True)
