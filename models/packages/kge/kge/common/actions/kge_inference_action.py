from heapq import heappush, heappushpop, nlargest
from typing import Any, NamedTuple, Protocol

import pandas as pd
import torch
from hakken_ml_toolkit.ml_utils.extras import TensorCreator
from torch.utils.data import DataLoader

from kge.common.entities.api import (
    KGEPredictRequest,
    KGEPredictResponse,
    KGEScoreIndexRequest,
    KGEScoreRequest,
    KGEScoreResponse,
)
from kge.data_processor.base import KGDataProcessor
from kge.models.base import KGEI

DEFAULT_SCORE = -1000.0


class ScoredTriple(NamedTuple):
    score: float
    subject: str
    relation: str
    object: str


class KGEInferenceActions(Protocol):
    @staticmethod
    def entity_pair_to_triples(
        subject_ids: list[str], relation_ids: list[str], object_ids: list[str]
    ) -> list[tuple[str, str, str]]:
        relation_ids_tuple = tuple(relation_ids)
        return [
            (s, r, o)
            for s, o in zip(subject_ids, object_ids, strict=False)
            for r in relation_ids_tuple
        ]

    @staticmethod
    @torch.no_grad()
    def predict(
        request: KGEPredictRequest,
        kge: KGEI,
        data_processing: KGDataProcessor,
        device: str | torch.device = "cpu",
    ) -> KGEPredictResponse:
        if request.relation_id_list is None:
            raise NotImplementedError()

        batch_size = 16192
        if request.inference_config is not None:
            batch_size = request.inference_config.get("batch_size", batch_size)

        relation_ids = request.relation_id_list

        relations_probs: list[list[float]] | None = None
        relations_scores: list[list[float]] = []

        num_pairs = len(request.subject_id_list)
        num_relations = len(relation_ids)

        if batch_size >= num_pairs:
            # Process all triples at once
            triples_list = KGEInferenceActions.entity_pair_to_triples(
                subject_ids=request.subject_id_list,
                relation_ids=request.relation_id_list,
                object_ids=request.object_id_list,
            )
            prediction_results = KGEInferenceActions.predict_batch(
                triples_list=triples_list,
                num_pairs=num_pairs,
                num_relations=num_relations,
                kge=kge,
                data_processing=data_processing,
                device=device,
            )
            relations_scores = prediction_results["relations_scores"]
            relations_probs = prediction_results["relations_probs"]
        else:
            for i in range(0, num_pairs, batch_size):
                subject_ids = request.subject_id_list[i : i + batch_size]
                object_ids = request.object_id_list[i : i + batch_size]
                triples_list = KGEInferenceActions.entity_pair_to_triples(
                    subject_ids=subject_ids,
                    relation_ids=relation_ids,
                    object_ids=object_ids,
                )
                batch_results = KGEInferenceActions.predict_batch(
                    triples_list=triples_list,
                    num_pairs=len(subject_ids),
                    num_relations=num_relations,
                    kge=kge,
                    data_processing=data_processing,
                    device=device,
                )
                relations_scores.extend(batch_results["relations_scores"])
                if batch_results["relations_probs"] is not None:
                    if relations_probs is None:
                        relations_probs = []
                    relations_probs.extend(batch_results["relations_probs"])

        return KGEPredictResponse.model_construct(
            relations_ids=relation_ids,
            relations_probs=relations_probs,
            relations_scores=relations_scores,
        )

    @staticmethod
    @torch.no_grad()
    def predict_batch(
        triples_list: list[tuple[str, str, str]],
        num_pairs: int,
        num_relations: int,
        kge: KGEI,
        data_processing: KGDataProcessor,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        sro_batch = data_processing.get_sro_batch(triples_list=triples_list, on_missing="ignore")

        sro_batch = sro_batch.to(device)
        all_valid = torch.all(sro_batch != -1)
        kge.to_device(device)
        kge.eval()
        if all_valid:
            score_batch = kge.score(sro_batch)
        else:
            valid_mask = (sro_batch != -1).all(dim=1)
            valid_mask = valid_mask.to(device)

            score_batch = torch.full(
                (len(triples_list), 1), DEFAULT_SCORE, dtype=torch.float, device=device
            )
            valid_sro = sro_batch[valid_mask]
            valid_scores = kge.score(valid_sro)
            score_batch[valid_mask] = valid_scores

        norm_score_batch: list[list[float]] | None = None
        if kge.has_scaler():
            norm_score_batch = (
                kge.normalize_scores(score_batch)
                .squeeze(-1)
                .view(num_pairs, num_relations)
                .tolist()
            )

        return {
            "relations_probs": norm_score_batch,
            "relations_scores": score_batch.squeeze(-1)
            .view(num_pairs, num_relations)
            .cpu()
            .tolist(),
        }

    @staticmethod
    def score(
        request: KGEScoreRequest,
        kge: KGEI,
        data_processing: KGDataProcessor,
        device: str | torch.device = "cpu",
    ) -> KGEScoreResponse:
        triples_list = request.facts_list

        sro_batch = data_processing.get_sro_batch(
            triples_list=triples_list, on_missing="ignore"
        ).to(device)

        return KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch, kge=kge, normalize=request.normalize, device=device
        )

    @staticmethod
    def score_from_tensor(
        sro_batch: torch.Tensor,
        kge: KGEI,
        normalize: bool = True,
        device: str | torch.device = "cpu",
    ) -> KGEScoreResponse:
        sro_batch = sro_batch.to(device)
        all_valid = torch.all(sro_batch != -1)
        num_facts = sro_batch.shape[0]
        kge.to_device(device)
        kge.eval()

        if all_valid:
            score_batch = kge.score(sro_batch)
        else:
            valid_mask = (sro_batch != -1).all(dim=1)
            valid_mask = valid_mask.to(device)

            score_batch = torch.full(
                (num_facts, 1), DEFAULT_SCORE, dtype=torch.float, device=device
            )
            valid_sro = sro_batch[valid_mask]
            valid_scores = kge.score(valid_sro)
            score_batch[valid_mask] = valid_scores

        normalized_scores_list: list[float] | None = None
        if normalize:
            normalized_scores_list = kge.normalize_scores(score_batch).flatten().tolist()

        if score_batch.shape[1] != 1:
            msg = "There should be only 1 column in the ScoreBatch"
            raise ValueError(msg)

        return KGEScoreResponse(
            scores_list=score_batch.flatten().tolist(),
            normalized_scores_list=normalized_scores_list,
        )

    @staticmethod
    def score_from_index(
        request: KGEScoreIndexRequest,
        kge: KGEI,
        device: str | torch.device = "cpu",
    ) -> KGEScoreResponse:
        triple_index_list = request.facts_index_list

        sro_batch = TensorCreator.long_tensor(triple_index_list, device=device)

        return KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch, kge=kge, normalize=request.normalize, device=device
        )

    @staticmethod
    def score_from_entity_list(
        entity_id_list: list[str],
        kge: KGEI,
        data_processing: KGDataProcessor,
        top_k: int = 100,
        device: str | torch.device = "cpu",
        batch_size=1024,
    ) -> pd.DataFrame:
        kge.to(device)
        kge.eval()

        relation_list = data_processing.relation_list()

        triples_list = []

        for subject_id in entity_id_list:
            for relation_id in relation_list:
                for object_id in entity_id_list:
                    if object_id == subject_id:
                        continue

                    triples_list.append((subject_id, relation_id, object_id))

        kge.to_device(device)
        kge.eval()

        triples_loader: DataLoader = DataLoader(
            triples_list,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )

        # Initialize min-heap to store top_k triples
        top_triples_heap: list[ScoredTriple] = []

        for triples_batch in triples_loader:
            sro_batch = data_processing.get_sro_batch(triples_list=triples_batch)
            score_batch = kge.score(sro_batch.to(device))

            for score, (s, r, o) in zip(score_batch, triples_batch, strict=False):
                scored_triple = ScoredTriple(
                    score=float(score.item()),
                    subject=s,
                    relation=r,
                    object=o,
                )

                if len(top_triples_heap) < top_k:
                    heappush(top_triples_heap, scored_triple)
                else:
                    heappushpop(top_triples_heap, scored_triple)

        top_k_triples: list[ScoredTriple] = nlargest(top_k, top_triples_heap)

        return pd.DataFrame(
            [
                {
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                    "score": t.score,
                }
                for t in top_k_triples
            ]
        )
