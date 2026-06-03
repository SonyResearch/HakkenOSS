from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd
from pydantic import BaseModel

# Define a TypeVar for the config that is a subclass of BaseModel
DataPrepareConfig = TypeVar("DataPrepareConfig", bound=BaseModel)
DataFinalizeConfig = TypeVar("DataFinalizeConfig", bound=BaseModel)


class DataProcessorBase(ABC, Generic[DataPrepareConfig, DataFinalizeConfig]):
    """
    Process a batch of hypotheses, cleaning and formatting to
    a parsable format for the next steps. Implements also the finalization.
    """

    @staticmethod
    @abstractmethod
    def prepare(df: pd.DataFrame, config: DataPrepareConfig) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def finalize(df: pd.DataFrame, config: DataFinalizeConfig) -> pd.DataFrame:
        pass
