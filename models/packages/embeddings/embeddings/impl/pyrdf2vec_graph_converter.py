from embeddings.core.contracts.graph_converter import IGraphConverter
from embeddings.core.entities.graph import Graph, LoadedGraph
from pyrdf2vec.graphs import kg
from pyrdf2vec.graphs.vertex import Vertex


class RDFConverter(IGraphConverter):
    def convert_graph(self, graph: LoadedGraph) -> Graph:
        pyrdf2vec_graph = kg.KG(mul_req=False)
        for subject, predicate, object in graph:
            s_v = Vertex(str(subject))
            o_v = Vertex(str(object))
            p_v = Vertex(str(predicate), predicate=True, vprev=s_v, vnext=o_v)
            pyrdf2vec_graph.add_vertex(s_v)
            pyrdf2vec_graph.add_vertex(p_v)
            pyrdf2vec_graph.add_vertex(o_v)
            pyrdf2vec_graph.add_edge(s_v, p_v)
            pyrdf2vec_graph.add_edge(p_v, o_v)

        return pyrdf2vec_graph
