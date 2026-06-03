from collections.abc import Callable

from kge.common.constants import InferenceTask

from .evaluate import run_evaluate
from .evaluate_mimic_kge import run_evaluate_mimic_kge
from .latency_profiler import run_latency_profiler
from .predict import run_predict, run_predict_from_file
from .save_embeddings import run_save_embeddings

INFERENCE_TASK_MAP: dict[str, Callable] = {
    InferenceTask.PREDICT: run_predict,
    InferenceTask.PREDICT_FROM_FILE: run_predict_from_file,
    InferenceTask.EVALUATE: run_evaluate,
    InferenceTask.EVALUATE_MIMIC_KGE: run_evaluate_mimic_kge,
    InferenceTask.LATENCY_PROFILER: run_latency_profiler,
    InferenceTask.SAVE_EMBEDDINGS: run_save_embeddings,
}
