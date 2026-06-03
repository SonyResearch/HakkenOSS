import pytest
from pydantic import ValidationError

from contextualization.core.entities.config.reference_reader import (
    NdjsonReferenceReaderConfig,
    ParquetReferenceReaderConfig,
)
from contextualization.core.entities.link import PublicationConceptLink
from contextualization.core.entities.publication import Publication
from contextualization.core.values.errors import ConfigurationError
from contextualization.impl.reference_reader.ndjson import NdjsonReferenceReader


@pytest.fixture
def config(ndjson_publications_directory, ndjson_publication_concept_links_directory):
    return NdjsonReferenceReaderConfig(
        publications_directory=ndjson_publications_directory,
        publication_concept_links_directory=ndjson_publication_concept_links_directory,
    )


class TestNdjsonReferenceReader:
    def test_config(
        self, ndjson_publications_directory, ndjson_publication_concept_links_directory, tmp_path
    ):
        ParquetReferenceReaderConfig(
            publications_directory=ndjson_publications_directory,
            publication_concept_links_directory=ndjson_publication_concept_links_directory,
        )
        ParquetReferenceReaderConfig(publications_directory=ndjson_publications_directory)
        ParquetReferenceReaderConfig(
            publication_concept_links_directory=ndjson_publication_concept_links_directory
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
        reader = NdjsonReferenceReader(config)

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
        reader = NdjsonReferenceReader(config)

        with pytest.raises(ConfigurationError):
            for _ in reader.iter_publications():
                pass

    def test_iter_publication_concept_links(self, config):
        reader = NdjsonReferenceReader(config)

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
        reader = NdjsonReferenceReader(config)

        with pytest.raises(ConfigurationError):
            for _ in reader.iter_publication_concept_links():
                pass
