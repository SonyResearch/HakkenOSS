import os
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor
from torch_geometric.utils import coalesce

VERSION_FILE = "version.txt"
DEFAULT_VERSIONS = {
    "KGE_VERSION": "latest",
    "DATA_VERSION": "latest",
}


def save_version(folder: str) -> None:
    """
    Save current environment versions into a version file.

    Args:
        folder: Directory where the version file will be written.
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    version_file = folder_path / VERSION_FILE

    versions = {
        "KGE_VERSION": os.getenv("KGE_VERSION", DEFAULT_VERSIONS["KGE_VERSION"]),
        "DATA_VERSION": os.getenv("DATA_VERSION", DEFAULT_VERSIONS["DATA_VERSION"]),
    }

    try:
        with version_file.open("w") as f:
            for key, value in versions.items():
                f.write(f"{key}={value}\n")

        logger.info(
            f"Saved version file at '{version_file}': "
            f"KGE_VERSION={versions['KGE_VERSION']}, DATA_VERSION={versions['DATA_VERSION']}"
        )
    except Exception as e:
        logger.error(f"Failed to save version file at '{version_file}': {e}")
        raise


def load_version(folder: str) -> bool:
    """
    Load versions from the version file and update environment variables.

    Args:
        folder: Directory where the version file is located.

    Returns:
        A dictionary with the loaded versions.
    """
    version_file = Path(folder) / VERSION_FILE
    versions = DEFAULT_VERSIONS.copy()
    success = False
    if version_file.exists():
        try:
            with version_file.open("r") as f:
                for line in f:
                    key, _, value = line.strip().partition("=")
                    if key in versions:
                        versions[key] = value or versions[key]

            logger.info(
                f"Loaded version file from '{version_file}': "
                f"KGE_VERSION={versions['KGE_VERSION']}, DATA_VERSION={versions['DATA_VERSION']}"
            )

            success = True
        except Exception as e:
            logger.error(f"Failed to load version file at '{version_file}': {e}")
            raise
    else:
        logger.warning(f"Version file not found at '{version_file}', using defaults: {versions}")

    os.environ.update(versions)

    return success


def remove_overlapping_edges(
    edges_to_exclude: Tensor,
    target_edge_index: Tensor,
    target_edge_labels: Tensor | None = None,
    device: str | torch.device | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Remove edges from target_edge_index that exist in edges_to_exclude.

    Args:
        edges_to_exclude: Edges that should be removed from targets with
            shape [2, num_exclude_edges].These are typically training/message-passing edges
            that we don't want to predict.
        target_edge_index: Target edges for prediction with shape [2, num_target_edges].
            These are the edges we want to make predictions on.
        target_edge_labels: Optional labels for target edges with shape [num_target_edges].
            These labels correspond to the target_edge_index edges.
        device: Device to place output tensors on. If None, uses the device of
            target_edge_index.

    Returns:
        Tuple containing:
        - filtered_target_edges: Target edges with overlapping edges removed
        - filtered_target_labels: Corresponding filtered target labels, or None if
        target_edge_labels was None
    """

    # Determine target device
    if device is None:
        device = target_edge_index.device

    if edges_to_exclude.numel() == 0:
        return target_edge_index, target_edge_labels

    if target_edge_index.numel() == 0:
        return target_edge_index, target_edge_labels

    # Ensure all input tensors are on the same device
    edges_to_exclude = edges_to_exclude.to(device)
    target_edge_index = target_edge_index.to(device)
    if target_edge_labels is not None:
        target_edge_labels = target_edge_labels.to(device)

    # Use hash-based approach to avoid coalesce issues with boolean tensors
    max_node_id = max(edges_to_exclude.max().item(), target_edge_index.max().item()) + 1

    # Create hash representations for efficient comparison
    exclude_hashes = edges_to_exclude[0] * max_node_id + edges_to_exclude[1]
    target_hashes = target_edge_index[0] * max_node_id + target_edge_index[1]

    # Find which target edges are NOT in the exclude set (vectorized)
    overlap_mask = torch.isin(target_hashes, exclude_hashes)
    keep_mask = ~overlap_mask

    # Filter edges and labels directly
    filtered_target_edges = target_edge_index[:, keep_mask]
    filtered_target_labels = (
        target_edge_labels[keep_mask] if target_edge_labels is not None else None
    )

    return filtered_target_edges, filtered_target_labels


def remove_overlapping_edges_2(
    edges_to_exclude: Tensor,
    target_edge_index: Tensor,
    target_edge_labels: Tensor | None = None,
    device: str | torch.device | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Remove edges from target_edge_index that exist in edges_to_exclude.

    Args:
        edges_to_exclude: Edges that should be removed from targets with
            shape [2, num_exclude_edges].
            These are typically training/message-passing edges that we don't want to predict.
        target_edge_index: Target edges for prediction with shape [2, num_target_edges].
            These are the edges we want to make predictions on.
        target_edge_labels: Optional labels for target edges with shape [num_target_edges].
            These labels correspond to the target_edge_index edges.
        device: Device to place output tensors on. If None, uses the device of
            target_edge_index.


    Returns:
        Tuple containing:
        - filtered_target_edges: Target edges with overlapping edges removed
        - filtered_target_labels: Corresponding filtered target labels, or None
            if target_edge_labels was None

    """

    if device is None:
        device = target_edge_index.device

    # Combine all edges to find overlaps
    all_edges: Tensor = torch.cat([edges_to_exclude, target_edge_index], dim=1)

    # Create edge attributes to track which edges are from which set
    exclude_mask: Tensor = torch.zeros(edges_to_exclude.size(1), dtype=torch.bool, device=device)
    target_mask: Tensor = torch.ones(target_edge_index.size(1), dtype=torch.bool, device=device)
    all_masks: Tensor = torch.cat([exclude_mask, target_mask])

    # Use coalesce to find unique edges and their indices
    unique_edges: Tensor
    unique_attrs: Tensor
    unique_edges, unique_attrs = coalesce(all_edges, all_masks, reduce="min")

    # Keep only edges that were originally from target_edge_index AND are unique
    keep_mask: Tensor = unique_attrs.bool()
    filtered_target_edges: Tensor = unique_edges[:, keep_mask]

    # Filter corresponding target labels efficiently (vectorized)
    filtered_target_labels: Tensor | None
    if target_edge_labels is not None:
        if filtered_target_edges.size(1) > 0:
            # Vectorized approach: find positions of filtered_target_edges in target_edge_index
            filtered_src = filtered_target_edges[0].unsqueeze(1)  # [num_filtered, 1]
            filtered_dst = filtered_target_edges[1].unsqueeze(1)  # [num_filtered, 1]

            target_src = target_edge_index[0].unsqueeze(0)  # [1, num_targets]
            target_dst = target_edge_index[1].unsqueeze(0)  # [1, num_targets]

            # Find matches: [num_filtered, num_targets]
            matches = (filtered_src == target_src) & (filtered_dst == target_dst)

            # Get the positions where matches occur
            match_positions = matches.nonzero(as_tuple=True)[1]  # Column indices
            filtered_target_labels = target_edge_labels[match_positions]
        else:
            filtered_target_labels = torch.tensor([], dtype=target_edge_labels.dtype, device=device)
    else:
        filtered_target_labels = None

    return filtered_target_edges, filtered_target_labels
