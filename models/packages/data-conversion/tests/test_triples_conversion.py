from __future__ import annotations

import itertools
import random
import unittest

import numpy as np
from parameterized import parameterized

from data_conversion import TriplesConverter
from data_conversion.triples.triple import Triple

test_num_entities = [10, 20, 100]
test_num_relations = [2, 5]
test_triples_factor = [1.0, 3.0]
test_format = ["list"]
test_use_embeddings = [True, False]
params = list(
    itertools.product(
        test_num_entities,
        test_num_relations,
        test_triples_factor,
        test_format,
        test_use_embeddings,
    )
)


class InvalidDimensionsError(ValueError):
    def __init__(self) -> None:
        super().__init__("Number of entities, relations, and triples must be greater than zero")


class TriplesFactory:
    @staticmethod
    def create_sample_triples_list(
        num_entities: int, num_relations: int, num_triples: int
    ) -> tuple[list[Triple], list[str], list[str]]:
        if num_entities <= 0 or num_relations <= 0 or num_triples <= 0:
            raise InvalidDimensionsError()

        entities = [f"entity-{i}" for i in range(1, num_entities + 1)]
        relations = [f"relation-{i}" for i in range(1, num_relations + 1)]
        tuple_triples: set[tuple[str, str, str]] = set()

        while len(tuple_triples) < num_triples:
            subject = str(random.choice(entities))
            predicate = str(random.choice(relations))
            obj = str(random.choice(entities))
            tuple_triple = (subject, predicate, obj)
            tuple_triples.add(tuple_triple)

        triples = [Triple.from_tuple(tuple_triple) for tuple_triple in tuple_triples]
        return triples, entities, relations


class TestTriplesConverter(unittest.TestCase):
    @parameterized.expand(params)
    def test_triples_list_to_networkx(
        self,
        num_entities: int,
        num_relations: int,
        triples_factor: float,
        format: str,
        use_embeddings: bool,
    ):
        num_triples = int(num_entities * triples_factor)

        triples, entities, relations = TriplesFactory.create_sample_triples_list(
            num_entities, num_relations, num_triples
        )
        if format == "array":
            raise NotImplementedError(
                "Test for input as a matrix of type "
                "`data_conversion.triples.types.NPArrayOfTriples` "
                "still needs to be implemented."
            )

        if use_embeddings:
            relation_emb_dict = {r: np.random.rand(10) for r in relations}
            entity_emb_dict = {e: np.random.rand(10) for e in entities}
            graph = TriplesConverter.to_networkx(
                triples,
                entities,
                relation_emb_dict=relation_emb_dict,
                entity_emb_dict=entity_emb_dict,
            )
        else:
            graph = TriplesConverter.to_networkx(triples, entities)

        for t_i_obj in triples:
            t_i = Triple.to_tuple(t_i_obj)

            self.assertTrue(graph.has_edge(t_i[0], t_i[2]))

            edge_i = graph[t_i[0]][t_i[2]]
            self.assertTrue(t_i[1] in edge_i["relation"])

        self.assertEqual(len(graph.nodes(data=False)), len(entities))
        for i, node in enumerate(graph.nodes(data=False)):
            self.assertEqual(node, entities[i])
            if use_embeddings:
                self.assertTrue(
                    np.array_equal(graph.nodes[node]["node_embedding"], entity_emb_dict[node])
                )

    @parameterized.expand(params)
    def test_triples_to_torch_geometric(
        self,
        num_entities: int,
        num_relations: int,
        triples_factor: float,
        format: str,
        use_embeddings: bool,
    ):
        num_triples = int(num_entities * triples_factor)

        triples, entities, relations = TriplesFactory.create_sample_triples_list(
            num_entities, num_relations, num_triples
        )

        if format == "array":
            raise NotImplementedError(
                "Test for input as a matrix of type "
                "`data_conversion.triples.types.NPArrayOfTriples` "
                "still needs to be implemented."
            )

        if use_embeddings:
            relation_emb_dict = {r: np.random.rand(10) for r in relations}
            entity_emb_dict = {e: np.random.rand(10) for e in entities}
            data = TriplesConverter.to_pyg(triples, entities, relation_emb_dict, entity_emb_dict)
        else:
            data = TriplesConverter.to_pyg(triples, entities)

        self.assertEqual(data.num_nodes, len(entities))
        self.assertLessEqual(data.num_edges, num_triples)

        for i, e in enumerate(entities):
            self.assertEqual(e, data.name[i])
            if use_embeddings:
                self.assertTrue(np.array_equal(data.node_embedding[i], entity_emb_dict[e]))
