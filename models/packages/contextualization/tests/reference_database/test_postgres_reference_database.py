import pytest

from contextualization.core.entities.config.reference_database import (
    PostgresReferenceDatabaseConfig,
)
from contextualization.core.values.errors import PublicationNotFoundError
from contextualization.impl.reference_database import PostgresReferenceDatabase


@pytest.fixture
def reference_database(postgres_connection_string) -> PostgresReferenceDatabase:
    config = PostgresReferenceDatabaseConfig(
        connection_string=postgres_connection_string,
        publication_table_name="publication",
        publication_concept_link_table_name="publication_concept",
    )
    return PostgresReferenceDatabase(config)


@pytest.mark.postgres
class TestPostgresReferenceDatabase:
    def test_get_publications(self, reference_database):
        publications = reference_database.get_publications(publication_ids=["id1", "id2"])
        assert publications[0].publication_id == "id1"
        assert publications[1].publication_id == "id2"

        with pytest.raises(PublicationNotFoundError):
            reference_database.get_publications(["NOT_EXIST"])

    def test_get_publication(self, reference_database):
        publication = reference_database.get_publication(publication_id="id1")
        assert publication.publication_id == "id1"

    @pytest.mark.parametrize("per_publication_max_size", [None, 1])
    def test_get_publication_concept_links_from_publication_ids(
        self, reference_database, per_publication_max_size
    ):
        publication_ids = ["id1", "id3"]

        links = reference_database.get_publication_concept_links_from_publication_ids(
            publication_ids, flatten=True, per_publication_max_size=per_publication_max_size
        )
        if per_publication_max_size:
            assert len(links) == 2
            assert "concept_id1" in [link.concept_id for link in links]
            assert "concept_id3" in [link.concept_id for link in links]
        else:
            assert len(links) == 3
            assert "concept_id1" in [link.concept_id for link in links]
            assert "concept_id2" in [link.concept_id for link in links]
            assert "concept_id3" in [link.concept_id for link in links]

        links_unflattened = reference_database.get_publication_concept_links_from_publication_ids(
            publication_ids, flatten=False, per_publication_max_size=per_publication_max_size
        )
        if per_publication_max_size:
            assert len(links_unflattened) == 2
            assert (
                sum(len(links_for_publication_id) for links_for_publication_id in links_unflattened)
                == 2
            )
            assert "concept_id1" in [link.concept_id for link in links_unflattened[0]]
            assert "concept_id3" in [link.concept_id for link in links_unflattened[1]]
        else:
            assert len(links_unflattened) == 2
            assert (
                sum(len(links_for_publication_id) for links_for_publication_id in links_unflattened)
                == 3
            )
            assert "concept_id1" in [link.concept_id for link in links_unflattened[0]]
            assert "concept_id2" in [link.concept_id for link in links_unflattened[0]]
            assert "concept_id3" in [link.concept_id for link in links_unflattened[1]]

    @pytest.mark.parametrize("per_concept_max_size", [None, 1])
    def test_get_publication_concept_links_from_concept_ids(
        self, reference_database, per_concept_max_size
    ):
        concept_ids = ["concept_id2", "concept_id3"]

        links = reference_database.get_publication_concept_links_from_concept_ids(
            concept_ids, flatten=True, per_concept_max_size=per_concept_max_size
        )
        if per_concept_max_size:
            assert len(links) == 2
            assert "id2" in [link.publication_id for link in links]
            assert "id3" in [link.publication_id for link in links]
        else:
            assert len(links) == 3
            assert "id1" in [link.publication_id for link in links]
            assert "id2" in [link.publication_id for link in links]
            assert "id3" in [link.publication_id for link in links]

        links_unflattened = reference_database.get_publication_concept_links_from_concept_ids(
            concept_ids, flatten=False, per_concept_max_size=per_concept_max_size
        )
        if per_concept_max_size:
            assert len(links_unflattened) == 2
            assert sum(len(links_for_concept) for links_for_concept in links_unflattened) == 2
            assert "id2" in [link.publication_id for link in links_unflattened[0]]
            assert "id3" in [link.publication_id for link in links_unflattened[1]]
        else:
            assert len(links_unflattened) == 2
            assert sum(len(links_for_concept) for links_for_concept in links_unflattened) == 3
            assert "id1" in [link.publication_id for link in links_unflattened[0]]
            assert "id2" in [link.publication_id for link in links_unflattened[0]]
            assert "id3" in [link.publication_id for link in links_unflattened[1]]
