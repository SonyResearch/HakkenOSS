from typing import TYPE_CHECKING

import pandas as pd

from contextualization.core.contracts.reference_database import ReferenceDatabase
from contextualization.core.entities.config.reference_database import (
    NdjsonReferenceDatabaseConfig,
)
from contextualization.core.entities.link import ConceptId, PublicationConceptLink
from contextualization.core.entities.publication import Publication, PublicationId
from contextualization.core.values.errors import PublicationNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence


class NdjsonReferenceDatabase(ReferenceDatabase[NdjsonReferenceDatabaseConfig]):
    def __init__(self, config: NdjsonReferenceDatabaseConfig):
        super().__init__(config)

        self.publication_df = pd.read_json(
            self.config.publications_path, lines=True, dtype={"pmid": str}
        )
        self.publication_concept_link_df = pd.read_json(
            self.config.publication_concept_links_path,
            lines=True,
            dtype={"pmid": str, "node_id": str},
        )

    def get_publications(self, publication_ids: "Sequence[PublicationId]") -> list[Publication]:
        filtered_df = self.publication_df[self.publication_df["pmid"].isin(publication_ids)]

        for publication_id in publication_ids:
            if publication_id not in filtered_df["pmid"].values:
                raise PublicationNotFoundError(publication_id=publication_id)

        publications_unordered = [Publication(**row.to_dict()) for _, row in filtered_df.iterrows()]
        publication_by_id = {pub.publication_id: pub for pub in publications_unordered}

        return [publication_by_id[publication_id] for publication_id in publication_ids]

    def _get_publication_concept_links_from_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        per_publication_max_size: int | None = None,
    ) -> list[list[PublicationConceptLink]]:
        grouped_dfs = [
            self.publication_concept_link_df[
                self.publication_concept_link_df["pmid"] == publication_id
            ]
            .merge(self.publication_df, on="pmid")
            .sort_values(["citations_count", "year"], ascending=[False, False])[
                :per_publication_max_size
            ]
            for publication_id in publication_ids
        ]
        return [
            [PublicationConceptLink(**row.to_dict()) for _, row in df_for_publication_id.iterrows()]
            for df_for_publication_id in grouped_dfs
        ]

    def _get_publication_concept_links_from_concept_ids(
        self, concept_ids: "Sequence[ConceptId]", per_concept_max_size: int | None = None
    ) -> list[list[PublicationConceptLink]]:
        grouped_dfs = [
            self.publication_concept_link_df[
                self.publication_concept_link_df["node_id"] == concept_id
            ]
            .merge(self.publication_df, on="pmid")
            .sort_values(["citations_count", "year"], ascending=[False, False])[
                :per_concept_max_size
            ]
            for concept_id in concept_ids
        ]
        return [
            [PublicationConceptLink(**row.to_dict()) for _, row in df_for_concept.iterrows()]
            for df_for_concept in grouped_dfs
        ]
