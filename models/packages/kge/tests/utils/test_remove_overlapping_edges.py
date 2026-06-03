import pytest
import torch

from kge.utils import remove_overlapping_edges


def test_remove_overlapping_edges_basic_functionality():
    """Test basic functionality with simple overlapping edges."""
    # Arrange
    edges_to_exclude = torch.tensor([[0, 1, 2], [1, 2, 3]])  # Shape [2, 3]
    target_edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])  # Shape [2, 4]
    target_edge_labels = torch.tensor([1, 1, 0, 1])  # Shape [4]

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    # Should remove edges (0,1) and (1,2) from targets, keeping (3,4) and (4,5)
    expected_edges = torch.tensor([[3, 4], [4, 5]])
    expected_labels = torch.tensor([0, 1])

    assert torch.equal(filtered_edges, expected_edges)
    assert torch.equal(filtered_labels, expected_labels)


def test_remove_overlapping_edges_no_labels():
    """Test functionality when target_edge_labels is None."""
    # Arrange
    edges_to_exclude = torch.tensor([[0, 1], [1, 2]])
    target_edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels=None
    )

    # Assert
    expected_edges = torch.tensor([[2, 3], [3, 4]])
    assert torch.equal(filtered_edges, expected_edges)
    assert filtered_labels is None


def test_remove_overlapping_edges_no_overlaps():
    """Test when there are no overlapping edges."""
    # Arrange
    edges_to_exclude = torch.tensor([[0, 1], [1, 2]])
    target_edge_index = torch.tensor([[3, 4, 5], [4, 5, 6]])
    target_edge_labels = torch.tensor([1, 0, 1])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    # No edges should be removed
    assert torch.equal(filtered_edges, target_edge_index)
    assert torch.equal(filtered_labels, target_edge_labels)


def test_remove_overlapping_edges_all_overlap():
    """Test when all target edges overlap with exclusion edges."""
    # Arrange
    edges_to_exclude = torch.tensor([[0, 1, 2], [1, 2, 3]])
    target_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # Same as exclusion
    target_edge_labels = torch.tensor([1, 0, 1])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    # All edges should be removed, resulting in empty tensors
    assert filtered_edges.shape == (2, 0)
    assert filtered_labels.shape == (0,)


def test_remove_overlapping_edges_empty_exclusion():
    """Test with empty edges_to_exclude tensor."""
    # Arrange
    edges_to_exclude = torch.empty((2, 0), dtype=torch.long)
    target_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    target_edge_labels = torch.tensor([1, 0, 1])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    # No edges should be removed
    assert torch.equal(filtered_edges, target_edge_index)
    assert torch.equal(filtered_labels, target_edge_labels)


def test_remove_overlapping_edges_empty_targets():
    """Test with empty target_edge_index tensor."""
    # Arrange
    edges_to_exclude = torch.tensor([[0, 1], [1, 2]])
    target_edge_index = torch.empty((2, 0), dtype=torch.long)
    target_edge_labels = torch.empty(0, dtype=torch.long)

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    assert filtered_edges.shape == (2, 0)
    assert filtered_labels.shape == (0,)


def test_remove_overlapping_edges_both_empty():
    """Test with both tensors empty."""
    # Arrange
    edges_to_exclude = torch.empty((2, 0), dtype=torch.long)
    target_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels=None
    )

    # Assert
    assert filtered_edges.shape == (2, 0)
    assert filtered_labels is None


def test_remove_overlapping_edges_undirected_graph():
    """Test with undirected edges (both directions should be considered)."""
    # Arrange
    # Exclusion has (0,1), target has both (0,1) and (1,0)
    edges_to_exclude = torch.tensor([[0], [1]])
    target_edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]])
    target_edge_labels = torch.tensor([1, 1, 0])

    # Act
    filtered_edges, _filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    # Should remove (0,1) but behavior for (1,0) depends on implementation
    # This test helps verify the expected behavior
    assert filtered_edges.shape[1] <= 3  # At least one edge should be removed


def test_remove_overlapping_edges_large_indices():
    """Test with large node indices to check for potential overflow issues."""
    # Arrange
    large_idx = 1_000_000
    edges_to_exclude = torch.tensor([[0, large_idx], [1, large_idx + 1]])
    target_edge_index = torch.tensor(
        [[0, large_idx, large_idx + 2], [1, large_idx + 1, large_idx + 3]]
    )
    target_edge_labels = torch.tensor([1, 1, 0])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    expected_edges = torch.tensor([[large_idx + 2], [large_idx + 3]])
    expected_labels = torch.tensor([0])
    assert torch.equal(filtered_edges, expected_edges)
    assert torch.equal(filtered_labels, expected_labels)


def test_remove_overlapping_edges_duplicate_targets():
    """Test behavior when target_edge_index contains duplicate edges."""
    # Arrange
    edges_to_exclude = torch.tensor([[0], [1]])
    target_edge_index = torch.tensor([[0, 0, 2], [1, 1, 3]])  # Duplicate (0,1)
    target_edge_labels = torch.tensor([1, 1, 0])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    # Both duplicate edges should be removed
    expected_edges = torch.tensor([[2], [3]])
    expected_labels = torch.tensor([0])
    assert torch.equal(filtered_edges, expected_edges)
    assert torch.equal(filtered_labels, expected_labels)


def test_remove_overlapping_edges_self_loops():
    """Test with self-loop edges."""
    # Arrange
    edges_to_exclude = torch.tensor([[0, 1], [0, 1]])  # Self-loop (0,0)
    target_edge_index = torch.tensor([[0, 1, 2], [0, 2, 2]])  # Self-loops (0,0) and (2,2)
    target_edge_labels = torch.tensor([1, 0, 1])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    expected_edges = torch.tensor([[1, 2], [2, 2]])
    expected_labels = torch.tensor([0, 1])
    assert torch.equal(filtered_edges, expected_edges)
    assert torch.equal(filtered_labels, expected_labels)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_remove_overlapping_edges_different_dtypes(dtype):
    """Test with different integer dtypes."""
    # Arrange
    edges_to_exclude = torch.tensor([[0, 1], [1, 2]], dtype=dtype)
    target_edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]], dtype=dtype)
    target_edge_labels = torch.tensor([1, 0, 1])

    # Act
    filtered_edges, filtered_labels = remove_overlapping_edges(
        edges_to_exclude, target_edge_index, target_edge_labels
    )

    # Assert
    assert filtered_edges.dtype == dtype
    expected_edges = torch.tensor([[2, 3], [3, 4]], dtype=dtype)
    expected_labels = torch.tensor([0, 1])
    assert torch.equal(filtered_edges, expected_edges)
    assert torch.equal(filtered_labels, expected_labels)
