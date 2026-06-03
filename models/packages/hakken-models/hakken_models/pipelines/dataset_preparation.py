import os

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
    KubernetesOrchestratorSettings,
)

from hakken_models.core.configs.dataset_preparator import DatasetPreparatorConfig
from hakken_models.steps.dataset.build_embeddings_step import (
    build_node_embeddings_step,
    build_relation_embeddings_step,
)
from hakken_models.steps.dataset.build_mappings import (
    build_domain_mapping,
    build_nodes_mapping,
    build_relation_mapping,
    build_timestamp_mapping,
)
from hakken_models.steps.dataset.build_relation_labels import (
    build_relation_labels_step,
)
from hakken_models.steps.dataset.build_tensors import build_tensor_step
from hakken_models.steps.dataset.concatenate_tensors import concatenate_dataframes
from hakken_models.steps.dataset.export_data_step import export_data_step
from hakken_models.steps.dataset.extract_metadata import extract_metadata_step
from hakken_models.steps.dataset.filter_and_split import filter_and_split_step
from hakken_models.steps.dataset.generate_data_quality_report import (
    generate_data_quality_report_step,
)
from hakken_models.steps.dataset.load_facts_df import load_facts_df
from hakken_models.steps.dataset.load_nodes_df import load_nodes_df

kubernetes_settings = KubernetesOrchestratorSettings(
    pod_settings={
        "env": [
            {"name": "ZENML_STORE_URL", "value": "http://mvp-zenml:8080"},
            {"name": "ZENML_STORE_TYPE", "value": "rest"},
        ]
    },
    orchestrator_pod_settings={
        "env": [
            {"name": "ZENML_STORE_URL", "value": "http://mvp-zenml:8080"},
            {"name": "ZENML_STORE_TYPE", "value": "rest"},
        ]
    },
)

docker_settings = DockerSettings(
    dockerfile="./Dockerfile.zenml",
    build_context_root="./",
    parent_image_build_config={
        "build_options": {
            "secret": [
                "id=codeartifact_username,env=UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME",  # From local env var
                "id=codeartifact_password,env=UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD",  # From local file
            ]
        },
    },
    image_tag="hakken-models",
    skip_build=False,  # Ensure ZenML doesn't skip the build step
    force_build=True,  # Forces a new build even if settings look the same
)


@pipeline(
    name="dataset_preparation",
    enable_cache=True,
    # settings={"orchestrator": kubernetes_settings, "docker": docker_settings},
)
def dataset_preparation_pipeline(config: DatasetPreparatorConfig) -> None:
    nodes_df_path = os.path.join(config.s3_raw_dir, "nodes_corrected.tsv")
    raw_nodes_df = load_nodes_df(nodes_df_path=nodes_df_path)

    facts_df_path = os.path.join(config.s3_raw_dir, "edges.tsv")
    raw_facts_df = load_facts_df(facts_df_path=facts_df_path)
    tuples_df = filter_and_split_step(
        facts_df=raw_facts_df,
        nodes_df=raw_nodes_df,
        allowed_relations=config.allowed_relations,
        temporal_partitions=config.temporal_partitions,
    )
    train_df = tuples_df[0]
    val_df = tuples_df[1]
    test_df = tuples_df[2]
    nodes_df = tuples_df[3]

    generate_data_quality_report_step.with_options(substitutions={"split_name": "train"})(
        id="generate_train_quality_report_step", df=train_df, num_samples=10_000
    )

    generate_data_quality_report_step.with_options(substitutions={"split_name": "val"})(
        id="generate_val_quality_report_step", df=val_df, num_samples=10_000
    )

    facts_df = concatenate_dataframes(train_df=train_df, val_df=val_df, test_df=test_df)

    domains_map_df = build_domain_mapping(nodes_df=nodes_df)
    nodes_map_df = build_nodes_mapping(nodes_df=nodes_df, domains_mapping_df=domains_map_df)
    relations_map_df = build_relation_mapping(facts_df=facts_df)
    timestamps_map_df = build_timestamp_mapping(facts_df=facts_df)

    extract_metadata_step(
        nodes_map_df=nodes_map_df,
        domains_map_df=domains_map_df,
        relations_map_df=relations_map_df,
        timestamps_map_df=timestamps_map_df,
    )

    train_np = build_tensor_step.with_options(substitutions={"split_name": "train"})(
        id="build_train_tensor_step",
        facts_df=train_df,
        relations_map_df=relations_map_df,
        nodes_map_df=nodes_map_df,
        timestamps_map_df=timestamps_map_df,
    )

    val_np = build_tensor_step.with_options(substitutions={"split_name": "val"})(
        id="build_val_tensor_step",
        facts_df=val_df,
        relations_map_df=relations_map_df,
        nodes_map_df=nodes_map_df,
        timestamps_map_df=timestamps_map_df,
    )

    test_np = build_tensor_step.with_options(substitutions={"split_name": "test"})(
        id="build_test_tensor_step",
        facts_df=test_df,
        relations_map_df=relations_map_df,
        nodes_map_df=nodes_map_df,
        timestamps_map_df=timestamps_map_df,
    )

    relation_labels = build_relation_labels_step(
        train_np=train_np,
        val_np=val_np,
        relations_map_df=relations_map_df,
    )
    train_relation_labels_np = relation_labels[0]
    val_relation_labels_np = relation_labels[1]

    node_embeddings_np = build_node_embeddings_step(
        nodes_map_df=nodes_map_df,
        vector_table_name=config.node_vectors_table_name,
        pg_connection_string=config.pg_connection_string,
    )

    relation_embeddings_np = build_relation_embeddings_step(
        relations_map_df=relations_map_df,
        vector_table_name=config.relation_vectors_table_name,
        pg_connection_string=config.pg_connection_string,
        id_column="relation_type",
    )

    export_data_step(
        domains_map_df=domains_map_df,
        nodes_map_df=nodes_map_df,
        relations_map_df=relations_map_df,
        timestamps_map_df=timestamps_map_df,
        train_np=train_np,
        val_np=val_np,
        test_np=test_np,
        target_root=config.output_dir,
        node_embeddings_np=node_embeddings_np,
        relation_embeddings_np=relation_embeddings_np,
        train_relation_labels_np=train_relation_labels_np,
        val_relation_labels_np=val_relation_labels_np,
    )
