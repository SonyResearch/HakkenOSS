import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph


from hakken_explainer.path_finder.impl.networkx import NetworkXPathFinder

load_dotenv()


kg_folder = Path(os.getenv("CACHED_DATA_FOLDER"))
source_id = "00064ef166585da2e7dbe45b6826affe"
target_id = "00069bed05f3c40e3e0f8b05cbe694e9"

source_id = "0d28423129e620ff4ef207fcf7df8b0d"
target_id = "71d4034793735c89bdcee46cc8572747"


device = "cuda"

kg = KnowledgeGraph.load(kg_folder)


source = kg.entity_mapping.id_to_index[source_id]
target = kg.entity_mapping.id_to_index[target_id]


facts_batch_all = kg.facts_dict["train"].to(device)

facts_batch: torch.Tensor = torch.unique(facts_batch_all, dim=0, sorted=False)
num_facts = facts_batch.shape[0]


logger.info(f"{facts_batch.shape} {facts_batch.device}")


logger.info(f"{source_id}[{source}] -> {target_id}[{target}]")

path_finder = NetworkXPathFinder(max_paths=100_000, undirected=True)

graph_folder = Path(os.getenv("GRAPH_CACHE_FOLDER"))
path_finder.setup(facts_batch, graph_folder)


k = None
tic = time.time()
paths = path_finder.find_paths(source=source, target=target, k=k)
delay = time.time() - tic
print(f"[{delay:.2f}] Found {len(paths)} paths of length {k} from {source} to {target}")


tic = time.time()
search_space = path_finder.convert_paths_to_search_space(paths)
delay = time.time() - tic
print(f"[{delay:.2f}] Search space is ready {search_space.shape}")
