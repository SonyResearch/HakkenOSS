from __future__ import annotations

import random
from typing import Any

import pandas as pd
from hakken_ml_toolkit.ml_utils.extras import TensorCreator
from hakken_ml_toolkit.ml_utils.function_timer import FunctionTimer, profile_method

from kge.common.actions.kge_inference_action import KGEInferenceActions
from kge.common.entities import (
    KGEPredictRequest,
    KGEPredictResponse,
    KGEScoreIndexRequest,
)
from kge.data_processor import KGDataProcessor
from kge.models import KGEI


class KGELatencyProfiler:
    """
    Benchmark latency for KGE models.
    """

    def __init__(self, data_processor: KGDataProcessor, model: KGEI) -> None:
        self.data_processor = data_processor
        self.model = model

        kg = self.data_processor.kg
        self.entity_id_list: list[str] = list(kg.entity_mapping.id_to_index.keys())
        self.all_relation_id_list: list[str] = list(kg.relation_mapping.id_to_index.keys())
        if not self.entity_id_list:
            msg = "entity_id_list is empty; cannot sample subjects/objects."
            raise ValueError(msg)

        self.num_entities = len(self.entity_id_list)
        self.num_relations = len(self.all_relation_id_list)

    def _sample_entity_pair_ids(self, batch_size: int) -> tuple[list[str], list[str]]:
        subjects = random.choices(self.entity_id_list, k=batch_size)
        objects = random.choices(self.entity_id_list, k=batch_size)
        return subjects, objects

    def run_predict(
        self,
        num_trials: int = 100,
        batch_size_list: list[int] | None = None,
        device_list: list[str] | None = None,
    ) -> pd.DataFrame:
        batch_size_list = batch_size_list or [32, 128, 1024]
        device_list = device_list or ["cpu", "cuda"]

        timer = FunctionTimer()
        self.data_processor.get_sro_batch = profile_method(
            timer, "predict/data_processor/get_sro_batch"
        )(self.data_processor.get_sro_batch)

        self.data_processor.kg.encode_facts = profile_method(
            timer, "predict/data_processor/encode_facts"
        )(self.data_processor.kg.encode_facts)

        self.model.score = profile_method(timer, "predict/model/score")(self.model.score)

        self.model.normalize_scores = profile_method(timer, "predict/model/normalize_scores")(
            self.model.normalize_scores
        )

        KGEInferenceActions.entity_pair_to_triples = profile_method(
            timer, "predict/entity_pair_to_triples"
        )(KGEInferenceActions.entity_pair_to_triples)

        TensorCreator.long_tensor = profile_method(timer, "predict/create_long_tensor")(
            TensorCreator.long_tensor
        )

        KGEInferenceActions.predict = profile_method(timer, "predict/action")(
            KGEInferenceActions.predict
        )

        KGEPredictResponse.model_construct = profile_method(timer, "predict/response_generation")(
            KGEPredictResponse.model_construct
        )

        # torch.Tensor.cpu = profile_method(
        #     timer, "predict/response_generation"
        # )(KGEPredictResponse.model_construct)

        data_list: list[dict[str, Any]] = []

        for device in device_list:
            for batch_size in batch_size_list:
                relation_id_list = self.all_relation_id_list

                timer.reset_records()
                for _ in range(num_trials):
                    s_ids, o_ids = self._sample_entity_pair_ids(batch_size)
                    request = KGEPredictRequest(
                        subject_id_list=s_ids,
                        object_id_list=o_ids,
                        relation_id_list=relation_id_list,
                        inference_config=None,
                    )
                    _ = KGEInferenceActions.predict(
                        request=request,
                        kge=self.model,
                        data_processing=self.data_processor,
                        device=device,
                    )

                data_list_i = self.get_data(
                    device=device,
                    batch_size=batch_size,
                    num_relation_types=len(relation_id_list),
                    timer=timer,
                )
                data_list.extend(data_list_i)
        return pd.DataFrame(data_list)

    def run_score(
        self,
        num_trials: int = 100,
        batch_size_list: list[int] | None = None,
        device_list: list[str] | None = None,
    ) -> pd.DataFrame:
        batch_size_list = batch_size_list or [32, 128, 1024]
        device_list = device_list or ["cpu", "cuda"]

        timer = FunctionTimer()
        self.data_processor.get_sro_batch = profile_method(timer, "data_processor/get_sro_batch")(
            self.data_processor.get_sro_batch
        )

        self.model.score = profile_method(timer, "model/score")(self.model.score)

        self.model.normalize_scores = profile_method(timer, "model/normalize_scores")(
            self.model.normalize_scores
        )

        KGEInferenceActions.score = profile_method(timer, "score_action")(KGEInferenceActions.score)

        data_list: list[dict[str, Any]] = []
        for device in device_list:
            for batch_size in batch_size_list:
                relation_id_list = self.all_relation_id_list

                timer.reset_records()
                for _ in range(num_trials):
                    triple_index_list = self._sample_triple_index_list(batch_size)
                    request = KGEScoreIndexRequest(
                        facts_index_list=triple_index_list, normalize=False
                    )
                    _ = KGEInferenceActions.score_from_index(
                        request=request, kge=self.model, device=device
                    )

                data_list_i = self.get_data(
                    device=device,
                    batch_size=batch_size,
                    num_relation_types=len(relation_id_list),
                    timer=timer,
                )
                data_list.extend(data_list_i)

        return pd.DataFrame(data_list)

    def _sample_triple_index_list(self, batch_size: int) -> list[list[int]]:
        subject_idx = random.choices(range(self.num_entities), k=batch_size)
        object_idx = random.choices(range(self.num_entities), k=batch_size)
        relation_idx = random.choices(range(self.num_relations), k=batch_size)

        return [[s, r, o] for s, r, o in zip(subject_idx, relation_idx, object_idx, strict=False)]

    def get_data(
        self,
        device: str,
        batch_size: int,
        num_relation_types: int,
        timer: FunctionTimer,
        num_facts: int = 100_000,
    ) -> list[dict[str, Any]]:
        cte = num_facts / batch_size
        data_list = []

        for action, action_stats in timer.get_stats(cte).items():
            data = {
                "batch_size": batch_size,
                "device": device,
                "num_relation_types": num_relation_types,
            }
            data["action"] = action

            data.update(**action_stats)
            data_list.append(data)

        return data_list
