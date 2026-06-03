import pytest
from pydantic import ValidationError

from contextualization.core.entities.config.reference_reader import (
    ParquetReferenceReaderConfig,
)
from contextualization.core.entities.link import PublicationConceptLink
from contextualization.core.entities.publication import Publication
from contextualization.core.values.errors import ConfigurationError
from contextualization.impl.reference_reader import ParquetReferenceReader


@pytest.fixture
def config(parquet_publications_directory, parquet_publication_concept_links_directory):
    return ParquetReferenceReaderConfig(
        publications_directory=parquet_publications_directory,
        publication_concept_links_directory=parquet_publication_concept_links_directory,
    )


class TestParquetReferenceReader:
    def test_config(
        self, parquet_publications_directory, parquet_publication_concept_links_directory, tmp_path
    ):
        ParquetReferenceReaderConfig(
            publications_directory=parquet_publications_directory,
            publication_concept_links_directory=parquet_publication_concept_links_directory,
        )
        ParquetReferenceReaderConfig(publications_directory=parquet_publications_directory)
        ParquetReferenceReaderConfig(
            publication_concept_links_directory=parquet_publication_concept_links_directory
        )

        with pytest.raises(ValidationError):
            ParquetReferenceReaderConfig()
        with pytest.raises(ValidationError):
            ParquetReferenceReaderConfig(publications_directory=tmp_path / "publications")
        with pytest.raises(ValidationError):
            ParquetReferenceReaderConfig(
                publication_concept_links_directory=tmp_path / "publication_concept_links"
            )

    def test_iter_publications(self, config):
        reader = ParquetReferenceReader(config)

        data = []
        for publication in reader.iter_publications():
            data.append(publication)
            assert isinstance(publication, Publication)
        assert len(data) == 3

        data = []
        for publication in reader.iter_publications(num_skips=1):
            data.append(publication)
        assert len(data) == 2

        data = []
        for publication in reader.iter_publications(num_skips=10000):
            data.append(publication)
        assert len(data) == 0

    def test_iter_publications_error(self, config):
        config.publications_directory = None
        reader = ParquetReferenceReader(config)

        with pytest.raises(ConfigurationError):
            for _ in reader.iter_publications():
                pass

    def test_iter_publication_concept_links(self, config):
        reader = ParquetReferenceReader(config)

        data = []
        for link in reader.iter_publication_concept_links():
            data.append(link)
            assert isinstance(link, PublicationConceptLink)
        assert len(data) == 5

        data = []
        for link in reader.iter_publication_concept_links(num_skips=1):
            data.append(link)
        assert len(data) == 4

        data = []
        for link in reader.iter_publication_concept_links(num_skips=10000):
            data.append(link)
        assert len(data) == 0

    def test_iter_publication_concept_links_error(self, config):
        config.publication_concept_links_directory = None
        reader = ParquetReferenceReader(config)

        with pytest.raises(ConfigurationError):
            for _ in reader.iter_publication_concept_links():
                pass
