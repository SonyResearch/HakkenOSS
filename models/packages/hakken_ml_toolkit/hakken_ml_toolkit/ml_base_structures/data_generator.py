import random

import numpy as np
import pandas as pd
import torch

from hakken_ml_toolkit.ml_base_structures import (
    KnowledgeGraph,
    Mapping,
)


class DummyDataGenerator:
    @staticmethod
    def scores(batch_size: int, num_entities: int) -> torch.Tensor:
        return torch.rand(batch_size, num_entities)

    @staticmethod
    def entity_batch(
        batch_size: int,
        num_entities: int = 1000,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)

        return torch.randint(0, num_entities, (batch_size,), generator=generator, device=device)

    @staticmethod
    def so_batch(
        batch_size: int,
        num_entities: int = 1000,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)

        return torch.randint(0, num_entities, (batch_size, 2), generator=generator, device=device)

    @staticmethod
    def ro_batch(
        batch_size: int,
        num_entities: int,
        num_relations: int,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)

        o = torch.randint(0, num_entities, (batch_size,), generator=generator, device=device)
        r = torch.randint(0, num_relations, (batch_size,), generator=generator, device=device)

        return torch.stack([r, o], dim=1)

    @staticmethod
    def sr_batch(
        batch_size: int,
        num_entities: int,
        num_relations: int,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)

        s = torch.randint(0, num_entities, (batch_size,), generator=generator, device=device)
        r = torch.randint(0, num_relations, (batch_size,), generator=generator, device=device)

        return torch.stack([s, r], dim=1)

    @staticmethod
    def sro_batch(
        batch_size: int,
        num_entities: int = 1000,
        num_relations: int = 100,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)

        result = torch.empty((batch_size, 3), dtype=torch.long, device=device)
        seen = set()

        count = 0

        while count < batch_size:
            # Generate a random triple
            sub = torch.randint(0, num_entities, (1,), generator=generator, device=device)
            rel = torch.randint(0, num_relations, (1,), generator=generator, device=device)
            obj = torch.randint(0, num_entities, (1,), generator=generator, device=device)

            # Convert to tuple for hashing
            triple = (sub.item(), rel.item(), obj.item())

            # If this triple is new, add it to the result
            if triple not in seen:
                seen.add(triple)
                result[count] = torch.tensor(triple)
                count += 1

        return result

    @staticmethod
    def facts_batch(  # noqa: PLR0913
        batch_size: int,
        num_entities: int = 1000,
        num_relations: int = 100,
        num_timestamps: int | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Generate a batch of temporal facts.

        When num_timestamps is provided, returns facts with shape [batch_size, 4]
        where each fact is [subject, relation, object, timestamp].
        When num_timestamps is None, returns facts with shape [batch_size, 3]
        where each fact is [subject, relation, object] (no timestamp).

        Args:
            batch_size: Number of facts to generate.
            num_entities: Maximum entity ID (exclusive).
            num_relations: Maximum relation ID (exclusive).
            num_timestamps: Maximum timestamp value (exclusive). If None, timestamps
                are not included in the output.
            device: Device to create tensors on.
            seed: Random seed for reproducibility. If None, uses default generator state.

        Returns:
            Tensor of shape [batch_size, 4] with dtype torch.long if num_timestamps
            is not None, otherwise shape [batch_size, 3] with dtype torch.long.
        """
        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)

        # Generate SRO triples
        sro = DummyDataGenerator.sro_batch(batch_size, num_entities, num_relations, device, seed)

        if num_timestamps is not None:
            timestamps = torch.randint(
                0, num_timestamps, (batch_size,), generator=generator, device=device
            )

            return torch.cat([sro, timestamps.unsqueeze(1)], dim=1)
        return sro

    @staticmethod
    def domains_mapping_dict(
        num_entities: int,
        num_domains: int,
        seed: int | None = None,
    ) -> dict:
        """
        Generate a polars DataFrame mapping node IDs to domain IDs.

        Args:
            num_entities: Number of entities to create mappings for.
            num_domains: Number of unique domains.
            seed: Random seed for reproducibility.

        Returns:
            polars DataFrame with columns 'node_id' and 'domain_id'.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        node_ids = list(range(num_entities))
        domain_ids = np.random.randint(0, num_domains, size=num_entities).tolist()

        return {
            "node_id": node_ids,
            "domain_id": domain_ids,
        }

    @staticmethod
    def entity_mapping(num_entities: int) -> Mapping:
        return Mapping(
            id_to_index={f"entity_{i}": i for i in range(num_entities)},
            index_to_id={i: f"entity_{i}" for i in range(num_entities)},
        )

    @staticmethod
    def relation_mapping(num_relations: int) -> Mapping:
        return Mapping(
            id_to_index={f"relation_{i}": i for i in range(num_relations)},
            index_to_id={i: f"relation_{i}" for i in range(num_relations)},
        )

    @staticmethod
    def relation_batch(
        batch_size: int,
        num_relations: int = 100,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)

        return torch.randint(0, num_relations, (batch_size,), generator=generator, device=device)

    @staticmethod
    def knowledge_graph_from_seed(
        seed: int,
        device: str | torch.device = "cpu",
        batch_size_values: list[int] | None = None,
    ) -> KnowledgeGraph:
        if batch_size_values is None:
            batch_size_values = [4, 8, 16, 64, 512, 4096]
        random.seed(seed)

        batch_size = random.choice(batch_size_values)
        num_entities = random.choice([100, 200, 500, 1000, 2000])
        num_relations = random.choice([10, 20, 50])

        return DummyDataGenerator.knowledge_graph(
            batch_size=batch_size,
            num_entities=num_entities,
            num_relations=num_relations,
            device=device,
            seed=seed,
        )

    @staticmethod
    def random_string_ids(list_length: int, vocab_size: int, prefix: str = "elem") -> list[str]:
        """
        Generate a list of string IDs of length `list_length`, sampled with replacement
        from {f"{prefix}_{i}" for i in range(vocab_size)}.
        """
        id_pool = [f"{prefix}_{i}" for i in range(vocab_size)]
        return random.choices(id_pool, k=list_length)

    @staticmethod
    def facts_df(
        batch_size: int,
        num_entities: int = 1000,
        num_relations: int = 100,
        num_timestamps: int | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        data = {}

        data["subject"] = DummyDataGenerator.random_string_ids(
            batch_size, vocab_size=num_entities, prefix="entity"
        )
        data["relation_type"] = DummyDataGenerator.random_string_ids(
            batch_size, vocab_size=num_relations, prefix="rel"
        )
        data["object"] = DummyDataGenerator.random_string_ids(
            batch_size, vocab_size=num_entities, prefix="entity"
        )
        if num_timestamps is not None:
            data["timestamp"] = np.random.randint(0, num_timestamps, size=batch_size).tolist()
        return pd.DataFrame(data)

    @staticmethod
    def knowledge_graph(
        batch_size: int,
        num_entities: int = 1000,
        num_relations: int = 100,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> KnowledgeGraph:
        sro_batch = DummyDataGenerator.sro_batch(
            batch_size, num_entities, num_relations, device, seed
        )

        facts_dict = {}
        facts_dict["all"] = sro_batch
        facts_dict["train"] = sro_batch
        facts_dict["val"] = sro_batch
        facts_dict["test"] = sro_batch

        entity_mapping = DummyDataGenerator.entity_mapping(num_entities)
        relation_mapping = DummyDataGenerator.relation_mapping(num_relations)

        return KnowledgeGraph(
            facts_dict=facts_dict,
            num_entities=num_entities,
            num_relations=num_relations,
            entity_mapping=entity_mapping,
            relation_mapping=relation_mapping,
        )
