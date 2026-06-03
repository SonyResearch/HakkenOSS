from dotenv import load_dotenv

from contextualization.api.container import ApiConfig, ContextualizationContainer
from contextualization.impl.publication_scorer.coverage import CoveragePublicationScorer
from contextualization.impl.reference_database import NdjsonReferenceDatabase
from contextualization.impl.retriever.lookup import LookupRetriever


class TestContextualizationContainer:
    def test_from_env(self, contextualization_container_config_env_path):
        load_dotenv(contextualization_container_config_env_path, override=True)

        container = ContextualizationContainer()
        container.config.from_pydantic(ApiConfig())  # type: ignore

        container.wire(modules=[__name__], packages=["contextualization"])

        assert isinstance(container.reference_database(), NdjsonReferenceDatabase)
        assert isinstance(container.publication_scorer(), CoveragePublicationScorer)
        assert container.context_summarizer() is None
        assert isinstance(container.retriever(), LookupRetriever)
