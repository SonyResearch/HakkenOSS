import numpy as np
import polars as pl
from loguru import logger
from zenml import log_metadata, step
from zenml.io import fileio


def write_df(df: pl.DataFrame, path: str, mode: str = "wb") -> None:
    """Save Polars DataFrame as Parquet to any ZenML-supported filesystem."""
    with fileio.open(path, mode) as f:
        df.write_parquet(f)


def write_npy(array: np.ndarray, path: str) -> None:
    """Save NumPy array as .npy to any ZenML-supported filesystem."""
    with fileio.open(path, "wb") as f:
        np.save(f, array)


@step
def export_data_step(
    domains_map_df: pl.DataFrame,
    nodes_map_df: pl.DataFrame,
    relations_map_df: pl.DataFrame,
    timestamps_map_df: pl.DataFrame,
    train_np: np.ndarray,
    val_np: np.ndarray,
    test_np: np.ndarray,
    target_root: str,
    node_embeddings_np: np.ndarray | None = None,
    relation_embeddings_np: np.ndarray | None = None,
    train_relation_labels_np: np.ndarray | None = None,
    val_relation_labels_np: np.ndarray | None = None,
) -> None:
    """
    Export datasets to a clean S3 structure that can be used WITHOUT ZenML.

    Layout:
        <base_prefix>/<dataset_name>/<dataset_version>/
            mappings/
                domains_map.parquet
                nodes_map.parquet
                relations_map.parquet
                timestamps_map.parquet

            tensors/
                train.npy
                val.npy
                test.npy

            embeddings/          (when provided)
                nodes.npy
                relations.npy

    Relation label arrays are saved alongside split tensors when provided:
        tensors/train_relation_labels.npy
        tensors/val_relation_labels.npy
    """
    import s3fs

    mappings_root = f"{target_root}/mappings"
    tensors_root = f"{target_root}/tensors"

    logger.info(f"📤 Exporting dataset to S3: {target_root}")

    fs = s3fs.S3FileSystem()

    with fs.open(f"{mappings_root}/domains_map.parquet", "wb") as f:
        domains_map_df.write_parquet(f)
    with fs.open(f"{mappings_root}/nodes_map.parquet", "wb") as f:
        nodes_map_df.write_parquet(f)
    with fs.open(f"{mappings_root}/relations_map.parquet", "wb") as f:
        relations_map_df.write_parquet(f)
    with fs.open(f"{mappings_root}/timestamps_map.parquet", "wb") as f:
        timestamps_map_df.write_parquet(f)
    with fs.open(f"{tensors_root}/train.npy", "wb") as f:
        np.save(f, train_np)
    with fs.open(f"{tensors_root}/val.npy", "wb") as f:
        np.save(f, val_np)
    with fs.open(f"{tensors_root}/test.npy", "wb") as f:
        np.save(f, test_np)

    metadata: dict[str, object] = {
        "export_root": target_root,
        "domains_map_path": f"{mappings_root}/domains_map.parquet",
        "nodes_map_path": f"{mappings_root}/nodes_map.parquet",
        "relations_map_path": f"{mappings_root}/relations_map.parquet",
        "timestamps_map_path": f"{mappings_root}/timestamps_map.parquet",
        "train_tensor_path": f"{tensors_root}/train.npy",
        "val_tensor_path": f"{tensors_root}/val.npy",
        "test_tensor_path": f"{tensors_root}/test.npy",
        "train_tensor_shape": list(train_np.shape),
        "val_tensor_shape": list(val_np.shape),
        "test_tensor_shape": list(test_np.shape),
    }

    if node_embeddings_np is not None and relation_embeddings_np is not None:
        embeddings_root = f"{target_root}/embeddings"
        with fs.open(f"{embeddings_root}/nodes.npy", "wb") as f:
            np.save(f, node_embeddings_np)
        with fs.open(f"{embeddings_root}/relations.npy", "wb") as f:
            np.save(f, relation_embeddings_np)

        metadata.update(
            {
                "node_embeddings_path": f"{embeddings_root}/nodes.npy",
                "relation_embeddings_path": f"{embeddings_root}/relations.npy",
                "node_embeddings_shape": list(node_embeddings_np.shape),
                "relation_embeddings_shape": list(relation_embeddings_np.shape),
            }
        )
        logger.info(
            f"📦 Exported embeddings: nodes={node_embeddings_np.shape}, "
            f"relations={relation_embeddings_np.shape}"
        )

    if train_relation_labels_np is not None and val_relation_labels_np is not None:
        with fs.open(f"{tensors_root}/train_relation_labels.npy", "wb") as f:
            np.save(f, train_relation_labels_np)
        with fs.open(f"{tensors_root}/val_relation_labels.npy", "wb") as f:
            np.save(f, val_relation_labels_np)

        metadata.update(
            {
                "train_relation_labels_path": f"{tensors_root}/train_relation_labels.npy",
                "val_relation_labels_path": f"{tensors_root}/val_relation_labels.npy",
                "train_relation_labels_shape": list(train_relation_labels_np.shape),
                "val_relation_labels_shape": list(val_relation_labels_np.shape),
            }
        )
        logger.info(
            f"📦 Exported relation labels: train={train_relation_labels_np.shape}, "
            f"val={val_relation_labels_np.shape}"
        )

    log_metadata(metadata=metadata)

    logger.info("🎯 Export completed successfully.")
