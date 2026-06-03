import logging

import networkx as nx
import pandas as pd
from hakken_ml_toolkit.ml_utils.networkx import NetworkXUtils
from pydantic import BaseModel

from hypgen_pipeline.core.contracts.entity_filter import FilterBase
from hypgen_pipeline.core.values.defaults import (
    NODE_PAIR_OCIDS_COLUMN_DEFAULT,
)
from hypgen_pipeline.core.values.exceptions import MissingReferenceKgError

logger = logging.getLogger(__name__)


class PathLengthFilterConfig(BaseModel):
    node_pair_ocids_column: str = NODE_PAIR_OCIDS_COLUMN_DEFAULT
    reference_kg: nx.Graph | None = None
    min_path_length: int | None = None
    max_path_length: int | None = None
    include_extrema: bool = False

    class Config:
        arbitrary_types_allowed = True


class PathLengthFilter(FilterBase[PathLengthFilterConfig]):
    """
    Filters hypotheses by selecting those where the shortest path connecting
    the nodes is within a certain length range.
    """

    @staticmethod
    def filter(df: pd.DataFrame, config: PathLengthFilterConfig) -> pd.DataFrame:
        # Variables renaming
        node_pair_ocids_column = config.node_pair_ocids_column
        reference_kg = config.reference_kg
        min_path_length = config.min_path_length
        max_path_length = config.max_path_length
        include_extrema = config.include_extrema

        if reference_kg is None:
            raise MissingReferenceKgError

        # Add information with the shortest path to the dataframe
        df["shortest_path_length"] = df.apply(
            lambda x: NetworkXUtils.get_shortest_path_length(
                graph=reference_kg,
                source=x[node_pair_ocids_column][0],
                target=x[node_pair_ocids_column][1],
                include_extrema=include_extrema,
            ),
            axis=1,
        )

        # If filtering is on, filter by target interval [min_path_length, max_path_length]
        if config.min_path_length is not None and config.max_path_length is not None:
            logger.info(
                f"Filtering paths outside of the interval: [{min_path_length},{max_path_length}]"
            )
            df = df.loc[
                (df["shortest_path_length"] >= min_path_length)
                & (df["shortest_path_length"] <= max_path_length)
            ].reset_index(drop=True)
        elif config.max_path_length is not None:
            logger.info(f"Filtering paths shorter than: {max_path_length}")
            df = df.loc[df["shortest_path_length"] <= max_path_length].reset_index(drop=True)
        elif config.min_path_length is not None:
            logger.info(f"Filtering paths longer than: {min_path_length}")
            df = df.loc[df["shortest_path_length"] >= min_path_length].reset_index(drop=True)

        return df
