from strenum import StrEnum


class StageType(StrEnum):
    """
    Enumerates the types of layer connection mechanisms available in GNN architectures.

    Attributes:
        SKIPSUM: Implements skip connections by adding the input features to the output
                 of each GNN layer, similar to ResNet architectures. This helps with
                 training deeper networks by mitigating the vanishing gradient problem.
    """

    SKIPSUM = "skipsum"


class PoolingType(StrEnum):
    """
    Enumerates the graph pooling operations available for aggregating node features
    into graph-level representations.

    Attributes:
        GLOBAL_ATT: Global attention pooling that computes attention weights for each
                    node and uses them for weighted pooling.
        MEAN: Global mean pooling that averages node features across each graph.
        MAX: Global max pooling that takes the maximum value for each feature dimension
             across all nodes in a graph.
        ADD: Global add/sum pooling that sums node features across each graph.
    """

    GLOBAL_ATT = "gatt"
    MEAN = "mean"
    MAX = "max"
    ADD = "add"
