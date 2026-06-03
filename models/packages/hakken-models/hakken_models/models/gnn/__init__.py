from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from torch_geometric.nn.models.basic_gnn import BasicGNN

from hakken_models.models.gnn.relational_sage import RelationalGraphSAGE
from hakken_models.registries.base import Registry


class GNNRegistry(Registry[BasicGNN]):
    pass


gnn_registry = GNNRegistry("GNN")

gnn_registry.register_class(GCN)
gnn_registry.register_class(GraphSAGE)
gnn_registry.register_class(GAT)
gnn_registry.register_class(RelationalGraphSAGE)

__all__ = ["gnn_registry"]
