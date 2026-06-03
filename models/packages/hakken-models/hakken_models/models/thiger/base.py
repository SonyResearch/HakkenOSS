from typing import cast

import torch
from loguru import logger
from torch import Tensor, nn

from hakken_models.core.configs.model import THiGERConfig
from hakken_models.core.entities.kg_data import KGData
from hakken_models.core.entities.temporal_kg_data import TemporalKGData
from hakken_models.models.gnn import gnn_registry
from hakken_models.models.nn import tx_registry


def find_elements_in_target(query: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """
    Finds which elements of query are in target and returns their indices in target.

    Args:
        query: Tensor of shape [num_elements_query] with no repeated elements
        target: Tensor of shape [num_elements_target] with no repeated elements

    Returns:
        mask: Boolean tensor of shape [num_elements_query] indicating which elements
              of query are in target
        indices_in_target: Tensor of shape [num_matches] with indices in target for
                          elements of query that are in target
    """
    mask = torch.isin(query, target)
    values_in_target = query[mask]

    # Sort target and get the original indices
    target_sorted, target_indices = torch.sort(target)

    # Find positions in sorted target
    pos_in_sorted = torch.searchsorted(target_sorted, values_in_target)

    # Map back to original indices in target
    indices_in_target = target_indices[pos_in_sorted]

    return mask, indices_in_target


class THiGER(nn.Module):
    """
    Temporal Hierarchical Graph Embedding Representation (THiGER) model for temporal KGs.

    THiGER combines Graph Neural Networks (GNNs) and Transformers to learn temporal and structural
    representations of entities in dynamic knowledge graphs. The model processes entity pairs across
    multiple timestamps, using GNNs to capture local graph structure and Transformers to model
    temporal dependencies.

    The model supports optional domain embeddings to enhance entity representations when entities
    belong to different domains or categories.

    Attributes:
        num_entities (int): Total number of unique entities in the knowledge graph.
        num_relations (int): Total number of relation types.
        num_timestamps (int): Total number of discrete timestamps in the temporal knowledge graph.
        num_domains (int | None): Total number of domains, if domain embeddings are used.
        entity_embedding_dim (int): Dimensionality of entity embeddings.
        relation_embedding_dim (int): Dimensionality of relation embeddings.
        domain_embedding_dim (int | None): Dimensionality of domain embeddings, if used.
        node_embedding_dim (int): Combined dimensionality of node embeddings (entity + domain).
        has_logits (bool): Whether the model includes a logits MLP for relation prediction.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_timestamps: int,
        gnn_name: str,
        gnn_kwargs: dict,
        transformer_name: str,
        transformer_kwargs: dict,
        has_logits: bool = True,
        entity_embedding_dim: int = 64,
        relation_embedding_dim: int = 64,
        domain_embedding_dim: int | None = None,
        num_domains: int | None = None,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_domains = num_domains
        self.num_timestamps = num_timestamps
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim

        if num_domains is not None and domain_embedding_dim is None:
            domain_embedding_dim = entity_embedding_dim

        self.domain_embedding_dim = domain_embedding_dim

        self.entity_embeddings = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=self.entity_embedding_dim,
        )

        self.node_embedding_dim = self.entity_embedding_dim
        if self.domain_embedding_dim is not None:
            self.node_embedding_dim += self.domain_embedding_dim

        self.domain_embeddings: nn.Embedding | None = None
        if num_domains is not None and self.domain_embedding_dim is not None:
            self.domain_embeddings = nn.Embedding(
                num_embeddings=num_domains,
                embedding_dim=self.domain_embedding_dim,
            )

        self.relation_embeddings = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=self.relation_embedding_dim,
        )
        self.pair_embedding_dim = self.node_embedding_dim * 2

        self.transformer = tx_registry.create(
            transformer_name,
            embedding_dim=self.pair_embedding_dim,
            max_seq_len=num_timestamps,
            **transformer_kwargs,
        )

        self.gnn = gnn_registry.create(
            gnn_name,
            in_channels=self.node_embedding_dim,
            out_channels=self.node_embedding_dim,
            **gnn_kwargs,
        )

        self._context_temporal_kg: TemporalKGData | None = None
        self._pair_embeddings_cache: Tensor | None = None

        self.has_logits = has_logits

        if self.has_logits:
            self.logits_mlp = nn.Linear(
                in_features=self.pair_embedding_dim,
                out_features=self.num_relations,
            )
        else:
            logger.debug("Model does not have logits, skipping logits MLP initialization.")

    @property
    def context_temporal_kg(self) -> TemporalKGData:
        """
        Get the context temporal knowledge graph.

        Returns:
            dict[int, KGData]: Dictionary mapping timestamp indices to KGData objects.

        Raises:
            ValueError: If the context temporal KG has not been set.
        """
        if self._context_temporal_kg is None:
            msg = "Context temporal KG has not been set."
            raise ValueError(msg)
        return self._context_temporal_kg

    def set_context_temporal_kg(self, kg_data: KGData) -> None:
        """
        Set the context temporal knowledge graph for the model.

        The context temporal KG is required for computing node
        embeddings and processing entity pairs during forward passes.

        Args:
            kg_data (KGData): The knowledge graph data to set as context. This will be
                converted to TemporalKGData internally.

        Note:
            Setting a new context temporal KG will invalidate any cached embeddings.
        """
        self._context_temporal_kg = TemporalKGData(kg_data)
        self.reset_cache()

    def clean_context_temporal_kg(self) -> None:
        """
        Clear the context temporal knowledge graph from memory.

        This method removes the stored temporal KG data, freeing memory. The model will need
        a new context temporal KG to be set before processing entity pairs.
        """
        self._context_temporal_kg = None

    def reset_cache(self) -> None:
        """
        Reset the internal cache for pair embeddings.

        Clears any cached embeddings from previous predictions. Useful when the context
        temporal KG changes or when you want to force recomputation.
        """
        self._pair_embeddings_cache = None

    def get_node_embeddings(self, entities: Tensor, domains: Tensor | None = None) -> Tensor:
        """
        Retrieve embeddings for specified entities, optionally concatenated with domain embeddings.

        Args:
            entities (Tensor): Entity indices, shape [num_entities].
            domains (Tensor | None): Optional domain indices corresponding to entities,
                                        shape [num_entities]. If provided and domain_embeddings
                                        is initialized, domain embeddings will be concatenated
                                        with entity embeddings.

        Returns:
            Tensor: Node embeddings, shape [num_entities, node_embedding_dim].
                        If domains are used, this is the concatenation of entity and domain
                        embeddings along the last dimension. Otherwise, returns only entity
                        embeddings with shape [num_entities, entity_embedding_dim].
        """

        entity_embeddings = self.entity_embeddings(entities)

        if self.domain_embeddings is not None and domains is not None:
            domain_embeddings = self.domain_embeddings(domains)
            return torch.cat([entity_embeddings, domain_embeddings], dim=-1)

        return cast(Tensor, entity_embeddings)

    def compute_node_embeddings_at_timestamp(
        self, timestamp_idx: int, entity_ids: Tensor
    ) -> Tensor:
        """
        Compute node embeddings for specified entities at a given timestamp.

        Args:
            timestamp_idx: The timestamp index to compute embeddings for.
            entity_ids: Global entity IDs to compute embeddings for,
                    shape [num_entities].

        Returns:
            Node embeddings tensor of shape [num_entities, node_embedding_dim],
            in the same order as input entity_ids.
        """

        kg_timestamp = self.context_temporal_kg.get_timestamp_data(
            timestamp_idx=timestamp_idx, relabel_nodes=True
        )

        # Get node data and domains for requested entities
        entity_node_data = self.context_temporal_kg.get_node_data(entity_ids, is_global=True)
        entity_domains = entity_node_data[:, 0]

        # Initialize embeddings for all requested entities (base embeddings without GNN)
        base_embeddings = self.get_node_embeddings(entities=entity_ids, domains=entity_domains)

        # Identify which requested entities are present in the timestamp subgraph
        # Compare using global IDs for consistency

        entities_in_timestamp_mask, n_id_indexes = find_elements_in_target(
            entity_ids, kg_timestamp.n_id
        )

        if entities_in_timestamp_mask.any():
            # Get domain information for subgraph nodes
            domains_timestamp = kg_timestamp.node_data[:, 0]

            # Compute GNN embeddings for all subgraph nodes
            base_embeddings_timestamp = self.get_node_embeddings(
                entities=kg_timestamp.n_id, domains=domains_timestamp
            )

            edge_types_timestamp = kg_timestamp.edge_attr[:, 0]
            relation_embeddings_timestamp = self.relation_embeddings(edge_types_timestamp)

            gnn_embeddings_timestamp = self.gnn(
                x=base_embeddings_timestamp,
                edge_index=kg_timestamp.edge_index,
                edge_attr=relation_embeddings_timestamp,
            )
            base_embeddings[entities_in_timestamp_mask] = gnn_embeddings_timestamp[n_id_indexes]

        return base_embeddings

    def compute_node_embeddings(self, entity_ids: Tensor | None = None) -> Tensor:
        """
        Compute temporal sequences of node embeddings for specified entities.

        This method processes entities across all timestamps in the context temporal KG,
        computing GNN-enhanced embeddings at each timestamp. For each timestamp, it extracts
        the subgraph, applies the GNN to capture local graph structure, and returns embeddings
        for the requested entities.

        Args:
            entity_ids (Tensor | None): Optional tensor of global entity IDs to compute
                embeddings for, shape [num_entities]. If None, computes embeddings for
                all entities in the context temporal KG.

        Returns:
            Tensor: Sequence of node embeddings across timestamps, shape
                [num_processed_timestamps, num_entities, node_embedding_dim].
                The first dimension corresponds to timestamps in the order they appear in
                context_temporal_kg, the second to entities, and the third to embedding dimensions.

        """

        if entity_ids is None:
            device = next(self.parameters()).device
            entity_ids = torch.arange(self.num_entities, device=device, dtype=torch.long)

        node_embeddings_sequence: list[Tensor] = []
        for timestamp_idx in self.context_temporal_kg.list_timestamps():
            embeddings = self.compute_node_embeddings_at_timestamp(
                timestamp_idx=timestamp_idx, entity_ids=entity_ids
            )

            node_embeddings_sequence.append(embeddings)

        # The shape is [num_processed_timestamps, batch_size, embedding_dim]
        return torch.stack(node_embeddings_sequence, dim=0)

    def compute_logits(
        self,
        entity_pair_batch: Tensor,
    ) -> Tensor:
        """
        Compute logits for entity pairs.

        This method computes relation prediction logits by passing pair embeddings through
        the logits MLP. The logits represent scores for each possible relation type.

        Args:
            entity_pair_batch (Tensor): Batch of entity pairs, shape [batch_size, 2].
                Each row contains [subject_entity_id, object_entity_id] as global entity IDs.

        Returns:
            Tensor: Logits for the entity pairs, shape [batch_size, num_relations].
                Higher values indicate higher probability of that relation type existing
                between the entity pair.

        Raises:
            ValueError: If the model was initialized with `has_logits=False`.
        """
        if not self.has_logits:
            msg = "Model does not have logits. Set `has_logits` to True in the config."
            raise ValueError(msg)
        # Forward pass through the transformer to get logits
        pair_embeddings = self.forward(entity_pair_batch)

        return self.logits_mlp.forward(pair_embeddings)

    def forward(
        self,
        entity_pair_batch: Tensor,
    ) -> Tensor:
        """
        Forward pass of the THiGER model.

        The method:
        1. Extracts unique entities from the batch
        2. Computes temporal sequences of embeddings for these entities
        3. Retrieves embeddings for subject and object entities in each pair
        4. Concatenates pair embeddings and processes through the Transformer

        Args:
            entity_pair_batch (Tensor): Batch of entity pairs, shape [batch_size, 2].
                Each row contains [subject_entity_id, object_entity_id] as global entity IDs.

        Returns:
            Tensor: Final embedding vectors for the entity pairs after Transformer processing,
                shape [batch_size, embedding_dim]. These embeddings can be used for relation
                prediction or other downstream tasks.

        Note:
            Requires context_temporal_kg to be set before calling. The method processes all
            timestamps in the context temporal KG to build temporal sequences.
        """

        entity_ids = cast("Tensor", torch.unique(entity_pair_batch))

        entity_seq_embeddings = self.compute_node_embeddings(entity_ids=entity_ids)

        subject_indices = self.context_temporal_kg.global_to_local(
            global_ids=entity_pair_batch[:, 0], n_id=entity_ids
        )
        object_indices = self.context_temporal_kg.global_to_local(
            global_ids=entity_pair_batch[:, 1], n_id=entity_ids
        )

        subject_seq_embeddings = entity_seq_embeddings[:, subject_indices, :]
        object_seq_embeddings = entity_seq_embeddings[:, object_indices, :]

        # [num_processed_timestamps, batch_size, 2*embedding_dim]
        pair_seq_embeddings = torch.cat([subject_seq_embeddings, object_seq_embeddings], dim=-1)
        # Transformer expects [B, N, D]
        pair_seq_embeddings = pair_seq_embeddings.transpose(0, 1)
        return self.transformer.forward(pair_seq_embeddings)

    def verify_entity_pairs(self, entity_pair_batch: Tensor) -> None:
        """
        Verify that entity_pair_batch is correctly formatted and contains valid entity IDs.

        This method performs comprehensive validation of the entity pair batch to ensure:
        - Correct tensor type and shape
        - Valid data type (integer/long)
        - Entity IDs within valid range
        - Context temporal KG is set (required for processing)

        Args:
            entity_pair_batch (Tensor): Batch of entity pairs to verify,
                expected shape [batch_size, 2]. Each row contains
                [subject_entity_id, object_entity_id] as global entity IDs.

        Raises:
            ValueError: If any validation check fails, with a descriptive error message.
            TypeError: If the input is not a Tensor.
        """
        # Type check
        if not isinstance(entity_pair_batch, Tensor):
            msg = f"entity_pair_batch must be a Tensor, got {type(entity_pair_batch)}."
            raise TypeError(msg)

        # Shape check: must be 2D with exactly 2 columns
        if entity_pair_batch.dim() != 2:
            msg = (
                f"entity_pair_batch must be 2D tensor with shape [batch_size, 2], "
                f"got shape {entity_pair_batch.shape} with {entity_pair_batch.dim()} dimensions."
            )
            raise ValueError(msg)

        if entity_pair_batch.size(1) != 2:
            msg = (
                f"entity_pair_batch must have 2 columns (subject, object), "
                f"got {entity_pair_batch.size(1)} columns."
            )
            raise ValueError(msg)

        # Non-empty batch check
        if entity_pair_batch.size(0) == 0:
            msg = "entity_pair_batch cannot be empty (batch_size must be > 0)."
            raise ValueError(msg)

        # Dtype check: must be integer type
        if entity_pair_batch.dtype != torch.long:
            msg = (
                f"entity_pair_batch must have integer dtype, got {entity_pair_batch.dtype}. "
                f"Consider converting with .long() or .to(torch.long)."
            )
            raise ValueError(msg)

        # Range check: entity IDs must be valid (0 <= id < num_entities)
        if torch.any(entity_pair_batch < 0):
            msg = "entity_pair_batch contains negative entity IDs. Entity IDs must be >= 0."
            raise ValueError(msg)

        if torch.any(entity_pair_batch >= self.num_entities):
            max_id = torch.max(entity_pair_batch).item()
            msg = (
                f"entity_pair_batch contains entity IDs >= num_entities ({self.num_entities}). "
                f"Found maximum ID: {max_id}. Entity IDs must be in [0, {self.num_entities - 1}]."
            )
            raise ValueError(msg)

        min_index = torch.min(entity_pair_batch).item()
        max_index = torch.max(entity_pair_batch).item()
        logger.debug(
            f"Entity pair batch verification passed: "
            f"shape={entity_pair_batch.shape}, dtype={entity_pair_batch.dtype}, "
            f"entity_id_range=[{min_index}, {max_index}]"
        )

    @classmethod
    def from_config(
        cls,
        config: THiGERConfig,
        num_entities: int,
        num_relations: int,
        num_timestamps: int,
        num_domains: int | None = None,
    ) -> "THiGER":
        return THiGER(
            num_entities=num_entities,
            num_relations=num_relations,
            num_timestamps=num_timestamps,
            gnn_name=config.gnn.name,
            gnn_kwargs=config.gnn.kwargs,
            transformer_name=config.transformer.name,
            transformer_kwargs=config.transformer.kwargs,
            has_logits=config.has_logits,
            entity_embedding_dim=config.entity_embedding_dim,
            relation_embedding_dim=config.relation_embedding_dim,
            domain_embedding_dim=config.domain_embedding_dim,
            num_domains=num_domains,
        )
