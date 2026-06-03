"""Encode publication vectors."""

import argparse

import torch.cuda
import torch.distributed as dist
from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer
from dependency_injector.wiring import Provide, inject
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pydantic_settings import BaseSettings, SettingsConfigDict

from contextualization.core.contracts.publication_encoder import (
    PublicationEncoder,
    PublicationEncoderToken,
)
from contextualization.core.contracts.publication_vector_database import (
    PublicationVectorDatabase,
    PublicationVectorDatabaseToken,
)
from contextualization.core.contracts.reference_reader import ReferenceReader, ReferenceReaderToken
from contextualization.core.entities.config.publication_encoder import PublicationEncoderConfig
from contextualization.core.entities.config.publication_vector_database import (
    PublicationVectorDatabaseConfig,
)
from contextualization.core.entities.config.reference_reader import ReferenceReaderConfig
from contextualization.initialize import (
    initialize_publication_encoder,
    initialize_publication_vector_database,
    initialize_reference_reader,
)


class EncodingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    reference_reader_config: ReferenceReaderConfig
    publication_encoder_config: PublicationEncoderConfig
    publication_vector_database_config: PublicationVectorDatabaseConfig


class EncodingContainer(DeclarativeContainer):
    config = providers.Configuration()

    reference_reader = providers.Singleton(
        initialize_reference_reader, config=config.reference_reader_config
    )
    publication_encoder = providers.Singleton(
        initialize_publication_encoder, config=config.publication_encoder_config
    )
    publication_vector_database = providers.Singleton(
        initialize_publication_vector_database, config=config.publication_vector_database_config
    )


@inject
def encode(
    reference_reader: ReferenceReader = Provide[ReferenceReaderToken],
    publication_encoder: PublicationEncoder = Provide[PublicationEncoderToken],
    publication_vector_database: PublicationVectorDatabase = Provide[
        PublicationVectorDatabaseToken
    ],
):
    publication_encoder.encode_and_store_to_db(
        reference_reader=reference_reader,
        publication_vector_database=publication_vector_database,
        skip_existing=True,
    )


def main():
    if torch.cuda.is_available():
        dist.init_process_group("nccl")
        torch.cuda.set_device(dist.get_rank())

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yaml-config-path", required=True, type=str)
    args = parser.parse_args()

    load_dotenv(".env")

    yaml_config = OmegaConf.to_object(OmegaConf.load(args.yaml_config_path))
    config = EncodingConfig.model_validate(yaml_config)

    container = EncodingContainer()
    container.config.from_pydantic(config)

    container.wire(modules=[__name__], packages=["contextualization"])

    encode()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
