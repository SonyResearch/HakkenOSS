from omegaconf import MISSING
from pydantic import BaseModel


class KGEConfig(BaseModel):
    name: str = "base"
    embedding_dim: int = MISSING
    num_entities: int = MISSING
    num_relations: int = MISSING
