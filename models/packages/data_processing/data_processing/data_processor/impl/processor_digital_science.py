from datetime import datetime
from pathlib import Path

from loguru import logger

from data_processing.data_processor.processor_base import DataProcessor
from data_processing.values import (
    NODE_DOMAIN_COLUMN,
    DataFrameType,
)


class DigitalScienceProcessor(DataProcessor[DataFrameType]):
    """Processor for Digital Science dataset with specific cleaning strategies"""

    def process(self) -> DataFrameType:
        """Digital Science-specific preparation strategies"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Load relations data the dataframe self.df
        self._load_relations_data()
        count = self.adapter.count_rows(self.df)
        logger.info(f"Relations data number of rows: {count}")
        self.adapter.show_head(self.df, n=10)

        # Load ontologies data into the dataframe self.df_ontologies
        self._load_ontologies_data()
        count = self.adapter.count_rows(self.df_ontologies)
        logger.info(f"Ontologies data number of rows: {count}")
        self.adapter.show_head(self.df_ontologies, n=10)

        return self.df

    def _load_relations_data(self):
        relations_config_list = self.config.data_files.relations
        for relations_cfg in relations_config_list:
            new_df = self.load_data(relations_cfg)
            self.df = new_df if not self.df else self.adapter.concat([self.df, new_df])

    def _load_ontologies_data(self):
        ontologies_config_list = self.config.data_files.ontologies
        for ontology_cfg in ontologies_config_list:
            new_df = self.load_data(ontology_cfg)
            # Add a column with the domain type
            ontology_name = Path(ontology_cfg.path).name
            if "csv_tree" in ontology_name:
                root_domain = ontology_name.replace("_hierarchy.csv_tree.csv", "")
            else:
                root_domain = ontology_name.replace("_hierarchy.csv", "")

            logger.info(f"Read {ontology_cfg.path} [{root_domain}]")

            def make_domain_func(domain: str):
                return domain

            new_df = self.adapter.apply_function_to_column(
                new_df, new_df.columns[0], NODE_DOMAIN_COLUMN, make_domain_func
            )
            self.df_ontologies = (
                new_df
                if not self.df_ontologies
                else self.adapter.concat([self.df_ontologies, new_df])
            )
