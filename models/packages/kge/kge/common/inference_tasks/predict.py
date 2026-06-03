from __future__ import annotations

from pathlib import Path
from time import perf_counter

from hakken_ml_toolkit.ml_utils import DSVUtils
from hakken_ml_toolkit.ml_utils.function_timer import FunctionTimer, profile_method
from loguru import logger
from omegaconf import DictConfig

from kge.common.actions.kge_inference_action import KGEInferenceActions
from kge.common.actions.kge_loader_action import KGEExperimentData
from kge.common.entities import KGEPredictRequest
from kge.data_processor import KGDataProcessor
from kge.models import KGEI


def run_predict_from_file(config: DictConfig, model: KGEI, data_processor: KGDataProcessor) -> None:
    df_entities = DSVUtils.read_dsv(file_path=Path(config.task.file_path), header=0)
    entity_list = list(df_entities["ocid_node"].values)

    df_top_k = KGEInferenceActions.score_from_entity_list(
        entity_id_list=entity_list,
        kge=model,
        data_processing=data_processor,
        top_k=10000,
        device=config.device,
    )
    df_top_pairs = df_top_k[["subject", "object"]].drop_duplicates()

    inference_folder = Path(config.output_dir)
    DSVUtils.write_dsv(
        df=df_top_k,
        file_path=inference_folder / "entity_model_hypotheses.csv",
        delimiter="\t",
    )
    DSVUtils.write_dsv(
        df=df_top_pairs,
        file_path=inference_folder / "entity_model_hypotheses_pair.csv",
        delimiter="\t",
    )


def run_predict(config: DictConfig, bundle: KGEExperimentData) -> None:
    request = KGEPredictRequest(
        subject_id_list=config.query.subject_id_list,
        object_id_list=config.query.object_id_list,
        relation_id_list=config.query.relation_id_list,
        inference_config=config.query.inference_config,
    )

    data_processor = bundle.data_processor
    model = bundle.model

    timer = FunctionTimer()
    data_processor.get_sro_batch = profile_method(timer, "data_processor/get_sro_batch")(
        data_processor.get_sro_batch
    )

    model.score = profile_method(timer, "model/score")(model.score)

    model.normalize_scores = profile_method(timer, "model/normalize_scores")(model.normalize_scores)

    tic = perf_counter()

    response = KGEInferenceActions.predict(
        request=request,
        kge=model,
        data_processing=data_processor,
        device=config.device,
    )
    delay = perf_counter() - tic
    logger.info(response)
    timer.summary()
    print(f"TOTAL: {delay:.6f}")
