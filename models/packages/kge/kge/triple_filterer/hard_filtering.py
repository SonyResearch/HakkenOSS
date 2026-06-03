from __future__ import annotations

from dataclasses import field

import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils

from kge.common.constants import TargetType
from kge.triple_filterer.base import TripleFiltererConfig, TripleFilterI


class HardTripleFilterConfig(TripleFiltererConfig):
    target: TargetType = TargetType.OBJECT
    filter_list: list[str] = field(default_factory=lambda: ["train"])


class HardTripleFilter(TripleFilterI[HardTripleFilterConfig]):
    def __init__(self, kg: KnowledgeGraph, config: HardTripleFilterConfig | None = None):
        self.config = config
        self.set_up(kg)

    def set_up(self, kg: KnowledgeGraph) -> None:
        sro_batch_list = []

        for key in self.config.filter_list:
            if key not in kg.facts_dict:
                continue
            sro_batch_list.append(kg.facts_dict[key].data)

        triples = PyTorchUtils.concat_tensors(sro_batch_list, dim=0)

        if self.config.target == "subject":
            self.ro_to_s: dict[tuple, set] = {}

            for s, r, o in triples.tolist():
                self.ro_to_s.setdefault((r, o), set()).add(s)

        elif self.config.target == "relation":
            self.so_to_r: dict[tuple, set] = {}

            for s, r, o in triples.tolist():
                self.so_to_r.setdefault((s, o), set()).add(r)

        elif self.config.target == "object":
            self.sr_to_o: dict[tuple, set] = {}

            for s, r, o in triples.tolist():
                self.sr_to_o.setdefault((s, r), set()).add(o)

    def compute_scores(
        self, sro_batch: torch.Tensor, scores: torch.Tensor, inplace: bool = True
    ) -> torch.Tensor:
        if not inplace:
            scores = scores.clone()

        if self.config.target == "subject":
            for i, (s, r, o) in enumerate(sro_batch.tolist()):
                pos_s = self.ro_to_s.get((r, o), set()) - {s}
                if pos_s:
                    scores[i, list(pos_s)] = float("-inf")

        elif self.config.target == "relation":
            for i, (s, r, o) in enumerate(sro_batch.tolist()):
                pos_r = self.so_to_r.get((s, o), set()) - {r}
                if pos_r:
                    scores[i, list(pos_r)] = float("-inf")

        elif self.config.target == "object":
            for i, (s, r, o) in enumerate(sro_batch.tolist()):
                pos_o = self.sr_to_o.get((s, r), set()) - {o}
                if pos_o:
                    scores[i, list(pos_o)] = float("-inf")

        return scores
