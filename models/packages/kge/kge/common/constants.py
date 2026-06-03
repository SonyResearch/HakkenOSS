from strenum import StrEnum


class KGEModelType(StrEnum):
    CONVE = "conve"


class InferenceTask(StrEnum):
    PREDICT = "predict"
    PREDICT_FROM_FILE = "predict_from_file"
    EVALUATE = "evaluate"
    EVALUATE_MIMIC_KGE = "evaluate_mimic_kge"
    LATENCY_PROFILER = "latency_profiler"
    SAVE_EMBEDDINGS = "save_embeddings"


class TargetType(StrEnum):
    SUBJECT = "subject"
    RELATION = "relation"
    OBJECT = "object"


class BaseFolderName(StrEnum):
    KGE = "kge"
    DATA_REPO = "data_repo"
    DATA_PROCESSOR = "data_processor"
    EMBEDDINGS = "embeddings"
