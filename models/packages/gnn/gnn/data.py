import torch
import torch_geometric.data as pygd

from gnn.common.domain import LongTensor2D


def prepare_fact_batch(
    fact_batch: LongTensor2D,
    num_nodes: int | None = None,
    num_relations: int | None = None,
) -> pygd.Data:
    """
    Converts a batch of knowledge graph facts into a PyTorch Geometric Data object.

    This function transforms knowledge graph triples (subject, relation, object) into a graph
    representation suitable for processing with PyTorch Geometric. The function handles
    non-temporal knowledge graphs and raises an error for temporal knowledge graphs.

    Args:
        fact_batch (LongTensor2D): A tensor of shape (batch_size, 3) containing knowledge
        graph facts. Each row represents a fact with [subject_id, relation_id, object_id].
            If shape is (batch_size, 4), it's assumed to be a temporal knowledge graph with
            an additional timestamp column, which is not supported.

        num_nodes (int | None, optional): The total number of nodes in the knowledge graph.
            If None, it's inferred as max(subject_id, object_id) + 1. Defaults to None.

        num_relations (int | None, optional): The total number of relation types in the
            knowledge graph. If None, it's inferred as max(relation_id) + 1.
            Defaults to None.

    Returns:
        pygd.Data: A PyTorch Geometric Data object containing:
            - edge_index: A tensor of shape [2, num_edges] with subject and object indices
            - edge_type: A tensor of shape [num_edges] with relation type indices
            - num_nodes: Total number of nodes in the graph
            - num_relations: Total number of relation types

    Raises:
        NotImplementedError: If the input is a temporal knowledge graph (4 columns)

    Example:
        >>> facts = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 2, 0]])
        >>> graph = prepare_knowledge_graph(facts)
        >>> print(graph)
        Data(edge_index=[2, 3], edge_type=[3], num_nodes=4, num_relations=3)
    """

    if fact_batch.shape[1] >= 4:
        msg = "Processing of Temporal Knowledge Graphs not implemented"
        raise NotImplementedError(msg)
    subjects = fact_batch[:, 0]
    relations = fact_batch[:, 1]
    objects = fact_batch[:, 2]

    if num_nodes is None:
        num_nodes = int(max(torch.max(subjects).item(), torch.max(objects).item())) + 1

    if num_relations is None:
        num_relations = int(torch.max(relations).item()) + 1

    edge_index = torch.stack([subjects, objects], dim=0)

    return pygd.Data(
        edge_index=edge_index,
        edge_type=relations,
        num_nodes=num_nodes,
        num_relations=num_relations,
    )
