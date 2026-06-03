from ast import literal_eval
from typing import Any

from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator

from hakken_models.core.entities.kg_data import KGData


def maybe_literal_eval(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def unflatten_dict(
    d: dict[str, Any],
    sep: str = "/",
) -> dict[str, Any]:
    """Reconstruct a nested dictionary from a flattened one."""

    result: dict[str, Any] = {}

    for flat_key, value in d.items():
        keys = flat_key.split(sep)
        current = result

        for key in keys[:-1]:
            current = current.setdefault(key, {})

        current[keys[-1]] = maybe_literal_eval(value)

    return result


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    """Flatten nested dictionary for hierarchical MLflow parameter logging"""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list | tuple):
            items.append((new_key, str(v)))
        elif v is not None:
            items.append((new_key, v))
    return dict(items)


def create_temporal_kg(
    num_timestamps: int,
    num_entities: int,
    num_relations: int,
    has_domains: bool = False,
    num_domains: int | None = None,
    seed: int | None = None,
) -> dict[int, KGData]:
    """Create temporal KG for testing using KGData.from_temporal_facts."""
    # Generate temporal facts using DummyDataGenerator
    num_facts_per_timestamp = max(50, num_entities * num_relations * 2)
    total_facts = num_timestamps * num_facts_per_timestamp

    facts = DummyDataGenerator.facts_batch(
        batch_size=total_facts,
        num_entities=num_entities,
        num_relations=num_relations,
        num_timestamps=num_timestamps,
        device="cpu",
        seed=seed,
    )

    # Create domains mapping if needed
    domains_mapping_df = None
    if has_domains and num_domains is not None:
        import polars as pl

        domains_dict = DummyDataGenerator.domains_mapping_dict(
            num_entities=num_entities,
            num_domains=num_domains,
            seed=seed,
        )

        domains_mapping_df = pl.DataFrame(domains_dict)

    return KGData.from_temporal_facts(
        facts=facts,
        domains_mapping_df=domains_mapping_df,
        num_nodes=num_entities,
        num_relations=num_relations,
        num_domains=num_domains,
    )
