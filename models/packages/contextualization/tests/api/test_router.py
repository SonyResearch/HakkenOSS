import pytest
from dotenv import load_dotenv

from contextualization.api.container import ApiConfig, ContextualizationContainer
from contextualization.api.entities import ContextualizationRequest
from contextualization.api.router import contextualize
from contextualization.core.entities.retrieval import RetrievalReturnType
from contextualization.core.entities.triple import Triple


@pytest.fixture
def container(contextualization_container_config_env_path) -> ContextualizationContainer:
    load_dotenv(contextualization_container_config_env_path, override=True)

    api_config = ApiConfig()  # type: ignore
    container = ContextualizationContainer()
    container.config.from_pydantic(api_config)
    return container


@pytest.fixture
def contextualization_request() -> ContextualizationRequest:
    return ContextualizationRequest(
        triples=[
            Triple(subject="concept_id1", relation="r1", object="concept_id2"),
            Triple(subject="concept_id2", relation="r2", object="concept_id3"),
        ],
        max_num_references=5,
        return_type=RetrievalReturnType.PUBLICATION,
    )


def test_contextualize(container, contextualization_request):
    container.wire(modules=[__name__], packages=["contextualization.api.router"])
    response = contextualize(request=contextualization_request)
    refs = response.references
    pubs = [ref.publication_info for ref in refs]
    pub_ids = [pub.publication_id for pub in pubs]

    assert len(pub_ids) == 3
    assert "id1" in pub_ids
    assert "id2" in pub_ids
    assert "id3" in pub_ids
