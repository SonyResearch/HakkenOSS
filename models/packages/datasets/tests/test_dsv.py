import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator

from datasets.data_repo.dsv import DSVKGConfig, DSVKGDataset


class TestDSVKGDataset:
    """Test suite for DSVKGDataset"""

    @pytest.fixture
    def sample_facts_data(self) -> pd.DataFrame:
        """Create sample facts data for testing"""
        return pd.DataFrame(
            {
                "subject_id": ["entity1", "entity2", "entity3"],
                "relation_id": ["rel1", "rel2", "rel1"],
                "object_id": ["entity2", "entity3", "entity1"],
            }
        )

    @pytest.fixture
    def temp_facts_file(self, sample_facts_data: pd.DataFrame) -> Generator[str, None, None]:
        """Create a temporary TSV file with sample data"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            sample_facts_data.to_csv(f.name, sep="\t", index=False)
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def basic_config(self, temp_facts_file: str) -> DSVKGConfig:
        """Create a basic DSVKGConfig for testing"""
        return DSVKGConfig(
            facts_file=temp_facts_file,
            facts_file_delimiter="\t",
            relation_column="relation_id",
            subject_column="subject_id",
            object_column="object_id",
            data_split_proportion_dict={"train": 0.8, "test": 0.2},
        )

    def test_init(self, basic_config: DSVKGConfig) -> None:
        """Test DSVKGDataset initialization"""
        dataset = DSVKGDataset(basic_config)
        assert dataset.config == basic_config
        assert dataset.name == "dsv"

    def test_get_config_class(self) -> None:
        """Test _get_config_class returns correct type"""
        assert DSVKGDataset._get_config_class() == DSVKGConfig

    @patch("datasets.data_repo.dsv.repo.DSVUtils.read_dsv")
    def test_load_from_database_basic(
        self,
        mock_read_dsv: MagicMock,
        basic_config: DSVKGConfig,
        sample_facts_data: pd.DataFrame,
    ) -> None:
        """Test basic _load_from_database functionality"""
        # Arrange
        mock_read_dsv.return_value = sample_facts_data

        # Act
        dataset = DSVKGDataset(basic_config)
        kg = dataset._load_from_database()

        # Assert
        assert isinstance(kg, KnowledgeGraph)

    @patch("datasets.data_repo.dsv.repo.DSVUtils.read_dsv")
    def test_load_from_database_with_timestamps(
        self,
        mock_read_dsv: MagicMock,
        temp_facts_file: str,
    ) -> None:
        """Test _load_from_database with timestamp data"""
        # Arrange
        config = DSVKGConfig(
            facts_file=temp_facts_file,
            subject_column="subject",
            relation_column="relation_type",
            object_column="object",
            timestamp_column="timestamp",
            data_split_proportion_dict={"train": 1.0},
        )
        num_timestamps = 4

        mock_facts_df = DummyDataGenerator.facts_df(
            batch_size=1000,
            num_entities=100,
            num_relations=4,
            num_timestamps=num_timestamps,
            seed=None,
        )
        mock_read_dsv.return_value = mock_facts_df

        # Act
        dataset = DSVKGDataset(config)
        kg = dataset._load_from_database()

        # Assert
        assert kg.num_timestamps == num_timestamps
