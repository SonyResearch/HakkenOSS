import torch

from .kg_data import KGData


class KGDataWithPreds(KGData):
    """
    KGData with all prediction fields required.

    This class is used for type checking to indicate that a KGData instance
    has all prediction fields (input_id, edge_label_index, edge_label) set and
    not None. It inherits all functionality from KGData but narrows the type
    of the prediction fields to be non-None.
    """

    input_id: torch.Tensor
    edge_label_index: torch.Tensor
    edge_label: torch.Tensor
