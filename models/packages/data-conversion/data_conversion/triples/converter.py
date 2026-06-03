from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import torch_geometric.data as pygd
from torch_geometric.utils import from_networkx

from data_conversion.triples.triple import Triple

if TYPE_CHECKING:
    from data_conversion.triples.types import NPArrayOfEntities, NPArrayOfTriples


class TriplesConverter:
    @staticmethod
    def to_networkx(
        triples: NPArrayOfTriples | list[Triple],
        entities: NPArrayOfEntities | list[str],
        relation_emb_dict: dict[str, np.ndarray] | None = None,
        entity_emb_dict: dict[str, np.ndarray] | None = None,
    ) -> nx.DiGraph:
        """
        Convert a list of triples into a NetworkX directed graph.
        """
        # Create an empty graph
        graph: nx.DiGraph = nx.DiGraph()

        nodes = [(n, {"name": n}) for n in entities]

        graph.add_nodes_from(nodes)

        # Iterate over the facts and add them as edges to the graph
        for _, triple_i in enumerate(triples):
            head_entity, relation_type, tail_entity = Triple.to_tuple(triple_i)

            if graph.has_edge(head_entity, tail_entity):
                graph[head_entity][tail_entity]["relation"].append(relation_type)
                if relation_emb_dict is not None:
                    graph[head_entity][tail_entity]["relation_embedding"].append(
                        relation_emb_dict[relation_type]
                    )
            else:
                attrs: dict[str, Any] = {"relation": [relation_type]}
                if relation_emb_dict is not None:
                    attrs["relation_embedding"] = [relation_emb_dict[relation_type]]

                graph.add_edge(head_entity, tail_entity, **attrs)
        if entity_emb_dict is not None:
            for node in graph.nodes():
                graph.nodes[node]["node_embedding"] = entity_emb_dict[node]
        return graph

    @staticmethod
    def to_pyg(
        triples: np.ndarray | list[Triple],
        entities: np.ndarray | list[str],
        relation_emb_dict: dict[str, np.ndarray] | None = None,
        entity_emb_dict: dict[str, np.ndarray] | None = None,
    ) -> pygd.Data:
        """
        Converts a list of triples into a PyTorch Geometric (PyG) Data object.
        """

        graph = TriplesConverter.to_networkx(triples, entities, relation_emb_dict, entity_emb_dict)

        return from_networkx(graph)
