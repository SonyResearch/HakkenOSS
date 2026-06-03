import torch
from kge.common.entities import KGPredictionSubgraph
from loguru import logger
from torch import Tensor
from torch_geometric.data import Batch
from tqdm import tqdm

from hakken_explainer.scores.base import ExplainerScore


class SufficientScore(ExplainerScore):
    """
    This scoring class aims to answer questions of the type: "What alone is enough to
    justify the prediction?" It measures how well individual explanation paths can
    independently support a target fact prediction, identifying the minimal sufficient
    evidence needed to justify model decisions in knowledge graph completion tasks.
    """

    @torch.no_grad()
    def score(
        self,
        target_fact: Tensor,
        candidate_paths: Tensor,
        batch_size: int = 1,
        num_hops: int = 2,
        normalize_by_original: bool = False,
    ) -> list[float]:
        if candidate_paths.numel() == 0:
            logger.warning("No candidate paths provided. Returning empty scores list.")
            return []
        device = candidate_paths.device
        model = self._prepare_model_and_device(device)

        original_score = 0.0
        if normalize_by_original:
            context_facts = self.get_context_kg(target_fact, k=num_hops)
            original_graph = KGPredictionSubgraph.from_facts(
                target_facts=target_fact, context_facts=context_facts.to(device)
            )

            original_score = model.score(original_graph).item()

        scores_list = []
        num_paths = candidate_paths.shape[0]

        for batch_start in tqdm(range(0, num_paths, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, num_paths)
            batch_explanations = candidate_paths[batch_start:batch_end]

            data_list = []
            for explanation in batch_explanations:
                subgraph = KGPredictionSubgraph.from_facts(
                    target_facts=target_fact, context_facts=explanation
                )
                data_list.append(subgraph)

            batch = Batch.from_data_list(data_list)
            batch_scores = model.score(batch)  # shape [batch_size, 1]

            if batch_scores.shape[1] != 1:
                msg = f"Expected batch_scores with shape [batch_size, 1], got {batch_scores.shape}"
                raise ValueError(msg)

            expl_score = batch_scores.flatten() - original_score
            scores_list.extend(expl_score.cpu().tolist())

        return scores_list
