from unittest.mock import MagicMock

import pandas as pd
from hakken_ml_toolkit.ml_base_structures import Mapping

from datasets.utils.dataframe import DataFrameUtils


def test_create_id_index_mapping():
    # Arrange
    data = {
        "subject_id": ["A", "B", "C"],
        "object_id": ["X", "Y", "Z"],
    }
    df = pd.DataFrame(data)

    # Act
    result = DataFrameUtils.create_id_index_mapping(df, ["subject_id", "object_id"])

    # Assert
    assert isinstance(result, Mapping)

    assert result.id_to_index == {"A": 0, "B": 1, "C": 2, "X": 3, "Y": 4, "Z": 5}
    assert result.index_to_id == {0: "A", 1: "B", 2: "C", 3: "X", 4: "Y", 5: "Z"}


def test_create_mappings_from_df():
    # Arrange
    data = {
        "subject_id": ["A", "B"],
        "relation_id": ["r1", "r2"],
        "object_id": ["X", "Y"],
    }
    df = pd.DataFrame(data)

    # Act
    mapping_dict = DataFrameUtils.create_mappings_from_df(
        df,
        subject_column="subject_id",
        relation_column="relation_id",
        object_column="object_id",
    )

    # Assert

    assert "entity" in mapping_dict
    assert "relation" in mapping_dict
    assert isinstance(mapping_dict["entity"], Mapping)
    assert isinstance(mapping_dict["relation"], Mapping)


def test_build_entity_domain_df():
    # Arrange
    data = {
        "subject_id": ["A", "B", "C"],
        "object_id": ["X", "Y", "Z"],
        "subject_domain": ["D1", "D2", "D3"],
        "object_domain": ["D4", "D5", "D6"],
    }
    df = pd.DataFrame(data)

    mock_entity_mapping = MagicMock()
    mock_entity_mapping.id_to_index = {"A": 0, "B": 1, "C": 2, "X": 3, "Y": 4, "Z": 5}
    # Act
    domain_mapping, entities_df = DataFrameUtils.build_entity_domain_df(
        facts_df=df,
        entity_mapping=mock_entity_mapping,
        subject_column="subject_id",
        object_column="object_id",
        subject_domain_column="subject_domain",
        object_domain_column="object_domain",
    )

    # Assert
    assert isinstance(domain_mapping, Mapping)

    assert isinstance(entities_df, pd.DataFrame)
    assert "entity_id" in entities_df.columns
    assert "entity_index" in entities_df.columns
    assert "domain_id" in entities_df.columns
    assert "domain_index" in entities_df.columns

    assert entities_df.shape[0] == 6
