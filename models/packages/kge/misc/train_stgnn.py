import torch
import networkx as nx
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch.nn as nn

from torch_geometric.nn.models import GAT
import numpy as np
import networkx as nx
from torch_geometric.utils import subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.utils.map import map_index


num_entities = 20
num_relations = 3
num_timestamps = 2
num_facts = 100
embedding_dim = 16


# Random integer generation for each column
subjects = torch.randint(0, num_entities, (num_facts, 1))
objects = torch.randint(0, num_entities, (num_facts, 1))
relations = torch.randint(0, num_relations, (num_facts, 1))
timestamps = torch.randint(0, num_timestamps, (num_facts, 1))

# Concatenate columns into a single tensor
facts = torch.cat([subjects, relations, objects, timestamps], dim=1)

# Drop duplicates
facts = torch.unique(facts, dim=0)

print(f"Number of unique facts: {facts.size(0)}")


edges: torch.Tensor = torch.unique(facts[:, [0, 2]], dim=0)
edges_np = edges.numpy()

edges_df = pd.DataFrame(edges_np)

subjects_np = edges_np[:, 0]

unique_subjects = np.unique(subjects_np)


k = 3


graph_partitions_tmp: list[list[np.ndarray]] = [[] for _ in range(k)]
for subject_idx in tqdm(unique_subjects):
    # Get boolean mask for current subject
    mask = subjects_np == subject_idx
    # Extract rows for this subject
    edges_i_np = edges_np[mask]
    subject_parts = np.array_split(edges_i_np, k)
    for i in range(k):
        graph_partitions_tmp[i].append(subject_parts[i])


graph_partitions: list[np.ndarray] = [
    np.concatenate(p_list) for p_list in graph_partitions_tmp
]

for graph in graph_partitions:
    print(graph.shape)


entity_embeddings = nn.Embedding(
    num_embeddings=num_entities, embedding_dim=embedding_dim
)
relation_embeddings = nn.Embedding(
    num_embeddings=num_relations, embedding_dim=embedding_dim
)
timestamp_embeddings = nn.Embedding(
    num_embeddings=num_timestamps, embedding_dim=embedding_dim
)

model = GAT(in_channels=embedding_dim, hidden_channels=embedding_dim, num_layers=2)

print(model)


data_list: list[Data] = []
for graph in graph_partitions:
    edge_index_np = graph.T
    unique_entities_np = np.unique(edge_index_np)

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    unique_entities = torch.tensor(unique_entities_np, dtype=torch.long)
    print(edge_index.shape)

    x = entity_embeddings(unique_entities)
    edge_index, node_mapping = map_index(
        edge_index.reshape(-1),
        unique_entities,
        max_index=num_entities,
        inclusive=True,
    )

    data = Data(x=x, edge_index=edge_index, node_mapping=node_mapping)
    data_list.append(data)
    edge_index = edge_index.view(2, -1)
    latent_embs = model.forward(x=x, edge_index=edge_index)
