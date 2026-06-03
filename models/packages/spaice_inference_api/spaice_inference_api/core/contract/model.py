from typing import Any

from pydantic import BaseModel
from typing_extensions import Protocol

ModelToken = "model"
ModelLoaderToken = "model_loader"


IModel = Any


class ModelLoadingOptions(BaseModel):
    path: str


class IModelLoader(Protocol):
    def load(self, options: ModelLoadingOptions) -> IModel:
        raise NotImplementedError()
