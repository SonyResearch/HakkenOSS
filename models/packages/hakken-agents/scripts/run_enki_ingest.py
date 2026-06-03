"""
Run the Enki pipeline with Hydra-composed configuration.

Loads config from configs/enki/enki.yaml (and defaults) via @hydra.main.
Override any config from the CLI, e.g.:

  uv run python scripts/run_enki_ingest.py
  uv run python scripts/run_enki_ingest.py entity_extractor.use_relevant_domains=false
  uv run python scripts/run_enki_ingest.py document.path=/path/to/doc.txt

  # Use local Ollama for entity (and optionally fact) extraction:
  uv run python scripts/run_enki_ingest.py entity_extractor/llm=ollama
  uv run python scripts/run_enki_ingest.py entity_extractor/llm=ollama fact_extractor/llm=ollama

Run from the hakken-agents package root (so config_path resolves and prompt paths work).
Ollama must be running (e.g. ollama run llama3.2). Override base_url with
entity_extractor/llm.base_url=... if needed.
"""

import asyncio
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from hydra.main import main as hydra_main
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from loguru import logger
from omegaconf import DictConfig

from hakken_agents.enki.config import EnkiConfig
from hakken_agents.enki.nodes.domain_resolver import DomainResolver
from hakken_agents.enki.nodes.entity_extractor import EntityExtractor
from hakken_agents.enki.nodes.entity_resolver import EntityResolver
from hakken_agents.enki.nodes.fact_extractor import FactExtractor
from hakken_agents.enki.nodes.fact_resolver import FactResolver
from hakken_agents.tools.document_parser import (
    DocumentParser,
    ParseDocumentConfig,
    ParseMethodType,
    ParserType,
    create_chunks,
)
from hakken_agents.tools.element_resolver import ElementResolver
from hakken_agents.tools.element_resolver.schemas import SimilaritySearchParam
from hakken_agents.utils.lightrag.kg.json_kv_impl import JsonKVStorage
from hakken_agents.utils.lightrag.kg.shared_storage import (
    initialize_pipeline_status,
    initialize_share_data,
)
from hakken_agents.utils.lightrag.utils import EmbeddingFunc

load_dotenv()

_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parent
_llm_cache_dir = _package_root / "cache"
_llm_cache_dir.mkdir(parents=True, exist_ok=True)
set_llm_cache(SQLiteCache(str(_llm_cache_dir / "llm_cache.db")))


# Absolute config dir so it works regardless of cwd
CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "enki")


async def _run(config: EnkiConfig) -> None:
    """Async pipeline: init cache, parse (with cache), chunk."""
    cache_dir = Path("./outputs")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Shared storage required by JsonKVStorage
    initialize_share_data(workers=1)
    await initialize_pipeline_status(workspace=config.workspace)

    # Dummy embedding func - parse cache does not use embeddings
    async def _dummy_embed(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 1), dtype=np.float32)

    parse_cache = JsonKVStorage(
        namespace="parse_cache",
        workspace=config.workspace,
        global_config={"working_dir": str(cache_dir), "embedding_batch_num": 10},
        embedding_func=EmbeddingFunc(embedding_dim=1, func=_dummy_embed),
    )
    await parse_cache.initialize()

    doc_parser_config = ParseDocumentConfig(
        file_path=config.document.path,
        output_dir="./outputs",
        parse_method=ParseMethodType.AUTO,
        parser=ParserType.MINERU,
        max_chunk_size=1000,
        lang=config.document.lang,
        device="cuda",
    )

    doc_parser = DocumentParser(config=doc_parser_config, parse_cache=parse_cache)

    entity_extractor = EntityExtractor.from_config(config.entity_extractor)
    chunks_resolver = ElementResolver.from_config(config.chunk_resolver)
    domains_resolver = DomainResolver.from_config(config.domain_resolver)
    entities_resolver = EntityResolver.from_config(config.entity_resolver)
    fact_extractor = FactExtractor.from_config(config.fact_extractor)
    fact_resolver = FactResolver.from_config(config.fact_resolver)

    if config.fact_extractor.preferred_relation_types:
        fact_resolver.seed_preferred_relations(config.fact_extractor.preferred_relation_types)

    if config.entity_extractor.allowed_domains:
        for domain in config.entity_extractor.allowed_domains:
            domain_doc = domains_resolver.resolve_domain(domain=domain)
            name = domain_doc.metadata.get("name", domain_doc.id)
            logger.info(f"Pre-inserted allowed domain: {domain_doc.id} - {name}")

    content_list, doc_id = await doc_parser.parse()

    # for content in content_list:
    #     print(content)
    logger.info(f"Doc ID: {doc_id}")

    chunks = create_chunks(content_list=content_list, max_chunk_size=1_000, doc_id=doc_id)

    previous_text = None

    for chunk in chunks:
        logger.info(chunk)
        chunk_doc = chunks_resolver.to_document(
            text=chunk.text,
            size=chunk.size,
            heading=chunk.levels.get("level_1", ""),
            doc_id=chunk.doc_id,
        )
        chunk_id = chunks_resolver.add(chunk_doc)

        if config.entity_extractor.use_relevant_domains:
            relevant_domains = domains_resolver.find_similar_elements(
                chunk.text, param=SimilaritySearchParam(k=10, threshold=0.5)
            )
            logger.info(f"relevant_domains: {relevant_domains}")
            if len(relevant_domains) == 0:
                relevant_domains = None
        else:
            relevant_domains = None

        if config.entity_extractor.allowed_domains:
            allowed_domains = "\n".join(config.entity_extractor.allowed_domains)
        else:
            allowed_domains = None

        user_variables = {}
        if relevant_domains is not None:
            user_variables["relevant_domains"] = relevant_domains
        if allowed_domains is not None:
            user_variables["allowed_domains"] = allowed_domains
        if previous_text is not None:
            user_variables["previous_text"] = previous_text

        result = entity_extractor.run(text=chunk.text, user_variables=user_variables)
        resolved_entities = []
        allowed_domains_list = config.entity_extractor.allowed_domains
        fallback_threshold = config.domain_resolver.fallback_threshold
        for entity in result.entities:
            print(entity)
            print("")
            domain_doc = domains_resolver.resolve_domain(
                domain=entity.domain,
                allowed_domains=allowed_domains_list,
                fallback_threshold=fallback_threshold,
            )
            if domain_doc is None:
                logger.warning(
                    f"Dropping entity {entity.name!r} (domain {entity.domain!r}): "
                    "no allowed domain match above threshold"
                )
                continue
            logger.info(f"resolved_domain: {domain_doc.id} - {domain_doc.metadata.get('name')}")

            entity_doc = entities_resolver.to_document(
                name=entity.name,
                description=entity.description,
                domain=domain_doc.metadata["name"],
                domain_id=domain_doc.id,
            )
            resolved_doc = entities_resolver.resolve_entity(entity_doc)
            logger.info(f"resolved_entity: {resolved_doc.id} - {resolved_doc.metadata.get('name')}")
            resolved_entities.append(resolved_doc)

        if resolved_entities:
            if config.fact_extractor.use_relevant_relation_types:
                rel_docs = fact_resolver.find_relevant_relation_types(
                    chunk.text, param=SimilaritySearchParam(k=10, threshold=0.5)
                )
                relevant_relation_types = (
                    "\n".join(doc.metadata["name"] for doc in rel_docs) if rel_docs else None
                )
                logger.info(f"relevant_relation_types:\n{relevant_relation_types}")
            else:
                relevant_relation_types = None

            if config.fact_extractor.preferred_relation_types:
                preferred_relation_types = "\n".join(config.fact_extractor.preferred_relation_types)
                logger.info(f"preferred_relation_types:\n{preferred_relation_types}")
            else:
                preferred_relation_types = None

            facts = fact_extractor.run(
                text=chunk.text,
                entities=resolved_entities,
                previous_text=previous_text,
                preferred_relation_types=preferred_relation_types,
                relevant_relation_types=relevant_relation_types,
            )
            for fact in facts.facts:
                logger.info(f"fact: {fact}")

            resolved_facts = fact_resolver.resolve_and_store_facts(
                facts=facts.facts,
                resolved_entities=resolved_entities,
                chunk_uuid=chunk_doc.id,
            )
            for fact_doc in resolved_facts:
                logger.info(f"stored: {fact_doc.page_content}")

        previous_text = chunk.text

    await parse_cache.finalize()


@hydra_main(version_base=None, config_path=CONFIG_PATH, config_name="enki")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: receives composed config, builds EnkiConfig, runs pipeline."""
    config = EnkiConfig.from_omegaconf(cfg)

    print(config.model_dump_json(indent=2))

    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
