from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from contextualization.core.contracts.reference_reader import ReferenceReader
from contextualization.core.entities.config.reference_reader import (
    ParquetReferenceReaderConfig,
)
from contextualization.core.entities.link import PublicationConceptLink
from contextualization.core.entities.publication import Publication
from contextualization.core.values.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterator


class ParquetReferenceReader(ReferenceReader[ParquetReferenceReaderConfig]):
    def __init__(self, config: ParquetReferenceReaderConfig):
        super().__init__(config)

    def iter_publications(self, num_skips: int = 0) -> "Iterator[Publication]":
        if not self.config.publications_directory:
            raise ConfigurationError("`publication_directory` is not set in the configuration")

        num_seen = 0

        file_paths = sorted(self.config.publications_directory.glob("*.parquet"))
        for file_path in file_paths:
            df = pd.read_parquet(file_path).replace({pd.NA: None, np.nan: None})

            if num_seen + len(df) < num_skips:
                num_seen += len(df)
                logger.info(
                    f"Skipped {file_path}, since num_seen: {num_seen}, num_skips: {num_skips}"
                )
                continue

            for _, row in df.iterrows():
                num_seen += 1
                if num_seen <= num_skips:
                    continue

                row_dict = row.to_dict()
                pmid = row_dict["pmid"]

                try:
                    year = row_dict["year"]
                    title = row_dict["title"]
                    abstract = row_dict["abstract"]
                    doi = row_dict["doi"]
                    citations_count = row_dict["citations_count"]
                    authors = row_dict["authors"]

                    publication = Publication(
                        publication_id=pmid,
                        year=year,
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        authors=authors,
                        citations_count=citations_count,
                    )
                    yield publication

                except KeyError as key:
                    logger.info(
                        f"Met KeyError while processing {pmid}, key: {key}; skipping the row"
                    )
                except ValueError as e:
                    logger.info(f"Got error while processing {pmid}; skipping the row")
                    logger.info(str(e))

    def iter_publication_concept_links(
        self, num_skips: int = 0
    ) -> "Iterator[PublicationConceptLink]":
        if not self.config.publication_concept_links_directory:
            raise ConfigurationError(
                "`publication_concept_links_directory` is not set in the configuration"
            )
        num_seen = 0

        file_paths = sorted(self.config.publication_concept_links_directory.glob("*.parquet"))
        for file_path in file_paths:
            df = pd.read_parquet(file_path).replace({pd.NA: None, np.nan: None})
            df = df[["pmid", "node_id"]]

            if num_seen + len(df) < num_skips:
                num_seen += len(df)
                logger.info(
                    f"Skipped {file_path}, since num_seen: {num_seen}, num_skips: {num_skips}"
                )
                continue

            for _, row in df.iterrows():
                num_seen += 1
                if num_seen <= num_skips:
                    continue

                row_dict = row.to_dict()
                publication_concept = PublicationConceptLink(**row_dict)
                yield publication_concept
