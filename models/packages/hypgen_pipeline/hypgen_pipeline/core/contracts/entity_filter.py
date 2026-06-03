from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd
from pydantic import BaseModel

# Define a TypeVar for the config that is a subclass of BaseModel
FilterConfig = TypeVar("FilterConfig", bound=BaseModel)


class FilterBase(ABC, Generic[FilterConfig]):
    """
    Process a batch of hypotheses with associated scores, filtering
    primarly based on some properties of the entities.
    """

    @staticmethod
    @abstractmethod
    def filter(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        pass
