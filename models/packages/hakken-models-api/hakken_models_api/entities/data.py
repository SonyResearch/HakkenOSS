from typing import Literal

from pydantic import BaseModel

FactType = tuple[str, str, str]
FactIndexType = tuple[int, int, int]


class EntityPairIndexRequest(BaseModel):
    subject_id_list: list[str]
    object_id_list: list[str]
    inference_config: dict | None = None


class EntityPairIndexResponse(BaseModel):
    subject_index_list: list[int]
    object_index_list: list[int]
    entity_pairs: list[tuple[int, int]]  # 2 x N or N x 2 depending on how you define it


class FactIndexRequest(BaseModel):
    facts_list: list[FactType]
    inference_config: dict | None = None


class FactIndexResponse(BaseModel):
    fact_index_list: list[FactIndexType]


class SampleFactsRequest(BaseModel):
    num_samples: int = 10
    splits: list[Literal["train", "val", "test"]] = ["train"]  # default: train split only


class SampleFactsResponse(BaseModel):
    facts_list: list[FactType]
