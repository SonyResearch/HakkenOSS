import yaml

from contextualization.delivery.encode import EncodingConfig, EncodingContainer
from contextualization.impl.publication_vector_database import InMemoryPublicationVectorDatabase
from contextualization.impl.reference_reader import ParquetReferenceReader


class TestEncodingContainer:
    def test_from_yaml(self, encoding_container_config_yaml_path):
        with open(encoding_container_config_yaml_path) as f:
            config = EncodingConfig.model_validate(yaml.safe_load(f))

        container = EncodingContainer()
        container.config.from_pydantic(config)

        container.wire(modules=[__name__], packages=["contextualization"])

        assert isinstance(container.reference_reader(), ParquetReferenceReader)
        assert isinstance(
            container.publication_vector_database(), InMemoryPublicationVectorDatabase
        )
