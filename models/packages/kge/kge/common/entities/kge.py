from dataclasses import dataclass

import torch
from datasets import DataRepositoryI
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph

from kge.data_processor import KGDataProcessor


@dataclass
class KGEForwardOutput:
    scores: torch.Tensor


@dataclass
class KGEDataBundle:
    data_repo: DataRepositoryI
    data_processor: KGDataProcessor
    kg: KnowledgeGraph
