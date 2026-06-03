import json
import time
from pathlib import Path
from typing import Any, cast

import requests
from loguru import logger

from data_processing.utils.common import chunk_list
from data_processing.values_pubtator import (
    PMID_TRACK_FILE,
    PUBTATOR_API_URL,
    PUBTATOR_CACHE_FILE,
    PUBTATOR_PUBLICATION_METADATA_CACHE_FILE,
    SLEEP_BETWEEN_CALLS,
)


def fetch_pubtator_data(unique_pmids: list[int], batch_size: int = 50) -> None:
    results = load_cache_pmids()
    pmids_to_process = [str(pmid) for pmid in unique_pmids if str(pmid) not in results]
    logger.info(f"PMIDS still to process: {len(pmids_to_process)}")

    for i, chunk in enumerate(chunk_list(pmids_to_process, batch_size), start=1):
        logger.info(f"Fetching batch {i}: {len(chunk)} PMIDs")
        records = fetch_pubtator_batch(chunk)
        if not records:
            logger.warning(f"Empty records for this batch of pmids: {chunk}")

        # Process response
        entries: list[dict[str, Any]] = []  # sentence-level records
        publication_metadata: list[dict[str, Any]] = []  # one per PMID

        entries, publication_metadata = process_pubtator_records(records)

        # Save
        save_cache_pubtator_data(entries)
        save_publication_metadata(publication_metadata)
        update_processed_pmids(chunk)

        time.sleep(SLEEP_BETWEEN_CALLS)  # ensures to not overload API


def fetch_pubtator_batch(pmid_list: list[str]) -> list[dict[str, Any]]:
    """Fetch JSON data for a batch of PMIDs via POST to PubTator 3 API."""
    if not pmid_list:
        return []

    # Build URL: append PMIDs comma-separated
    url = PUBTATOR_API_URL
    params = {"pmids": ",".join([str(pmid) for pmid in pmid_list])}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "PubTator3" in data:
            result = data["PubTator3"]
        else:
            logger.warning(f"Unexpected PubTator3 response keys: {list(data.keys())}")
            result = []

    except Exception as e:
        logger.error(f"⚠️ Failed to fetch batch of {len(pmid_list)} PMIDs: {e}")
        result = []

    return cast("list[dict[str, Any]]", result)


def process_pubtator_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process PubTator records into structured sentence entries and metadata."""
    entries: list[dict[str, Any]] = []
    publication_metadata: list[dict[str, Any]] = []

    for record in records:
        pmid = record.get("id", None)
        if not pmid:
            logger.warning(f"Empty pmid not found for record: {record}")
            continue

        record_entries, metadata = process_record(record, pmid)
        entries.extend(record_entries)
        publication_metadata.append(metadata)

    return entries, publication_metadata


def process_record(
    record: dict[str, Any], pmid: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Process a single PubTator record into entries and metadata."""
    passages = record.get("passages", [])
    if not passages:
        logger.warning(f"No passages found for record: {record}")

    entries: list[dict[str, Any]] = []

    pub_year, title, abstract_text = None, None, ""

    for passage in passages:
        p_type, text, year = extract_passage_info(passage)
        if p_type == "title":
            title = text.strip()
            pub_year = year
        elif p_type == "abstract":
            abstract_text += " " + text.strip()

        entities = extract_entities(passage)
        if text.strip() and entities:
            # Everything that is saved has a sentence with associated pmid and entities
            entries.append(
                {
                    "pmid": pmid,
                    "sentence": text,
                    "entities": entities,
                }
            )

    metadata = build_publication_metadata(pmid, title, abstract_text, pub_year)
    return entries, metadata


def extract_passage_info(passage: dict[str, Any]) -> tuple[str | None, str, str | None]:
    """Extract passage type, text, and year (if available)."""
    infons = passage.get("infons", {})
    p_type = infons.get("type")
    text = passage.get("text", "")
    year = infons.get("year")
    return p_type, text, year


def extract_entities(passage: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract entities (annotations) from a passage."""
    annotations = passage.get("annotations", [])
    entities = []

    for ann in annotations:
        entity = parse_annotation(ann)
        if entity:
            entities.append(entity)

    return entities


def parse_annotation(annotation: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single annotation into a structured entity."""
    info = annotation.get("infons") or {}
    text = annotation.get("text") or None
    loc = annotation.get("locations") or [{}]
    loc = loc[0]

    start = loc.get("offset")
    length = loc.get("length")
    end = start + length if isinstance(start, int) and isinstance(length, int) else None

    return {
        "name": info.get("name", text),
        "mention": text,
        "type": info.get("type", None),
        "id": info.get("identifier", None),
        "start": start,
        "end": end,
    }


def build_publication_metadata(
    pmid: str, title: str | None, abstract: str, year: str | None
) -> dict[str, Any]:
    """Assemble publication-level metadata."""
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract.strip(),
        "year": year,
    }


def save_cache_pubtator_data(entries: list[dict[str, Any]]) -> None:
    """Append entries to local JSONL cache file."""
    if not entries:
        return
    path = Path(PUBTATOR_CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)  # create parent directories if needed
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def save_publication_metadata(entries: list[dict[str, Any]]) -> None:
    """Save pubblication metadata separately to local JSONL cache file."""
    if not entries:
        return
    path = Path(PUBTATOR_PUBLICATION_METADATA_CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)  # create parent directories if needed
    with path.open("a", encoding="utf-8") as f_meta:
        for m in entries:
            f_meta.write(json.dumps(m, ensure_ascii=False) + "\n")


def load_cache_pmids() -> set[str]:
    """Load processed PMIDs from tracking file."""
    if not Path(PMID_TRACK_FILE).exists():
        return set()
    with open(PMID_TRACK_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def update_processed_pmids(pmids: list[str]) -> None:
    """Append new PMIDs to tracking file."""
    path = Path(PMID_TRACK_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(PMID_TRACK_FILE, "a", encoding="utf-8") as f:
        for pmid in pmids:
            f.write(f"{pmid}\n")
