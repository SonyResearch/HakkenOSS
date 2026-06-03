import json
from typing import Any
from unittest.mock import MagicMock, patch

from data_processing.utils.pubtator import (
    build_publication_metadata,
    extract_entities,
    extract_passage_info,
    fetch_pubtator_batch,
    fetch_pubtator_data,
    load_cache_pmids,
    parse_annotation,
    process_pubtator_records,
    process_record,
    save_cache_pubtator_data,
    save_publication_metadata,
    update_processed_pmids,
)

# ---------------- Test PMID tracking cache ----------------


def test_pmids_cache(tmp_path):
    track_file = tmp_path / "pmids.txt"

    # Patch the module variable to use the temporary file
    with patch("data_processing.utils.pubtator.PMID_TRACK_FILE", track_file):
        # Initially, no PMIDs
        pmids = load_cache_pmids()
        assert pmids == set()

        # Add PMIDs
        update_processed_pmids(["123", "456"])
        pmids = load_cache_pmids()
        assert pmids == {"123", "456"}

        # Add more PMIDs (including duplicates)
        update_processed_pmids(["456", "789"])
        pmids = load_cache_pmids()
        assert pmids == {"123", "456", "789"}


def test_pubtator_cache(tmp_path):
    # Prepare test file path
    cache_file = tmp_path / "cache.jsonl"
    entries = [{"pmid": "123", "sentence": "Test sentence", "entities": []}]

    # Patch the module variable to use temp file
    with patch("data_processing.utils.pubtator.PUBTATOR_CACHE_FILE", cache_file):
        save_cache_pubtator_data(entries)

    # Check file content
    lines = cache_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["pmid"] == "123"
    assert data["sentence"] == "Test sentence"


def test_publication_metadata_cache(tmp_path):
    # Prepare test file path
    metadata_file = tmp_path / "metadata.jsonl"
    entries = [{"pmid": "123", "title": "Test title", "abstract": "Abstract text", "year": 2020}]

    # Patch the module variable to use temp file
    with patch(
        "data_processing.utils.pubtator.PUBTATOR_PUBLICATION_METADATA_CACHE_FILE", metadata_file
    ):
        save_publication_metadata(entries)

    # Check file content
    lines = metadata_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["pmid"] == "123"
    assert data["title"] == "Test title"
    assert data["abstract"] == "Abstract text"
    assert data["year"] == 2020


def test_fetch_pubtator_batch_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"PubTator3": [{"id": "123"}, {"id": "456"}]}

    with patch(
        "data_processing.utils.pubtator.requests.get", return_value=mock_response
    ) as mock_get:
        result = fetch_pubtator_batch([123, 456])
        assert result == [{"id": "123"}, {"id": "456"}]
        mock_get.assert_called_once()
        # Ensure query string has comma-separated PMIDs
        called_params = mock_get.call_args.kwargs["params"]
        assert called_params["pmids"] == "123,456"


def test_fetch_pubtator_batch_http_error_logs():
    with patch("data_processing.utils.pubtator.requests.get", side_effect=Exception("HTTP error")):
        result = fetch_pubtator_batch(["123"])
        assert result == []


def test_fetch_pubtator_batch_empty_input():
    assert fetch_pubtator_batch([]) == []


def test_process_pubtator_records_basic() -> None:
    """Ensure proper transformation of PubTator3 records into entries and metadata."""
    records: list[dict[str, Any]] = [
        # Record missing id -> skipped
        {"passages": [], "infons": {}},
        # Record with id but no passages -> only metadata
        {"id": "123", "passages": [], "infons": {}},
        # Record with title + abstract + one entity
        {
            "id": "456",
            "passages": [
                {
                    "infons": {"type": "title", "year": "2020"},
                    "text": "Title of 456",
                    "annotations": [],
                },
                {
                    "infons": {"type": "abstract"},
                    "text": "This is a sentence.",
                    "annotations": [
                        {
                            "text": "gene",
                            "infons": {"type": "Gene", "identifier": "G1", "name": "gene"},
                            "locations": [{"offset": 0, "length": 4}],
                        }
                    ],
                },
            ],
        },
        # Record with title + abstract + two entities
        {
            "id": "789",
            "passages": [
                {
                    "infons": {"type": "title", "year": "2014"},
                    "text": "Title of 789",
                    "annotations": [],
                },
                {
                    "infons": {"type": "abstract"},
                    "text": "This is a sentence.",
                    "annotations": [
                        {
                            "text": "gene",
                            "infons": {"type": "Gene", "identifier": "G1", "name": "gene"},
                            "locations": [{"offset": 0, "length": 4}],
                        },
                        {
                            "text": "chemical",
                            "infons": {"type": "Chemistry", "identifier": "C1", "name": "chemical"},
                            "locations": [{"offset": 8, "length": 10}],
                        },
                    ],
                },
            ],
        },
    ]

    entries, metadata = process_pubtator_records(records)

    # --- Basic checks ---
    assert len(entries) == 2
    pmids = {e["pmid"] for e in entries}
    assert pmids == {"456", "789"}

    # --- Verify 456 ---
    record_456 = next(e for e in entries if e["pmid"] == "456")
    assert record_456["sentence"] == "This is a sentence."
    ent = record_456["entities"][0]
    assert ent == {
        "name": "gene",
        "mention": "gene",
        "type": "Gene",
        "id": "G1",
        "start": 0,
        "end": 4,
    }

    # --- Verify 789 ---
    record_789 = next(e for e in entries if e["pmid"] == "789")
    assert record_789["sentence"] == "This is a sentence."
    ent_2 = record_789["entities"][1]
    assert ent_2 == {
        "name": "chemical",
        "mention": "chemical",
        "type": "Chemistry",
        "id": "C1",
        "start": 8,
        "end": 18,
    }

    # --- Metadata ---
    pmids_meta = {m["pmid"] for m in metadata}
    assert pmids_meta == {"123", "456", "789"}

    meta_456 = next(m for m in metadata if m["pmid"] == "456")
    assert meta_456["year"] == "2020"
    assert meta_456["title"] == "Title of 456"
    assert meta_456["abstract"] == "This is a sentence."

    meta_789 = next(m for m in metadata if m["pmid"] == "789")
    assert meta_789["year"] == "2014"
    assert meta_789["title"] == "Title of 789"
    assert meta_789["abstract"] == "This is a sentence."


def test_extract_passage_info():
    passage = {"infons": {"type": "title", "year": "2023"}, "text": "Some text"}
    p_type, text, year = extract_passage_info(passage)
    assert (p_type, text, year) == ("title", "Some text", "2023")


def test_parse_annotation_valid():
    ann = {
        "infons": {"type": "Gene", "identifier": "G1", "name": "TP53"},
        "text": "TP53",
        "locations": [{"offset": 5, "length": 4}],
    }
    parsed = parse_annotation(ann)
    assert parsed["name"] == "TP53"
    assert parsed["mention"] == "TP53"
    assert parsed["start"] == 5
    assert parsed["end"] == 9
    assert parsed["type"] == "Gene"


def test_parse_annotation_missing_fields():
    ann = {"infons": {}, "text": "X"}
    parsed = parse_annotation(ann)
    assert parsed["mention"] == "X"
    assert parsed["name"] == "X"
    assert parsed["start"] is None
    assert parsed["end"] is None
    assert parsed["type"] is None
    assert parsed["id"] is None


def test_extract_entities_multiple():
    passage = {
        "annotations": [
            {
                "infons": {"type": "Gene", "identifier": "G1", "name": "TP53"},
                "text": "TP53",
                "locations": [{"offset": 5, "length": 4}],
            },
            {
                "infons": {"type": "Chem", "identifier": "C1", "name": "Aspirin"},
                "text": "Aspirin",
                "locations": [{"offset": 12, "length": 7}],
            },
            # malformed annotation: no location
            {
                "infons": {"type": "Disease", "identifier": "D1", "name": "Cancer"},
                "text": "Cancer",
                "locations": None,
            },
            # malformed annotation: missing infons
            {
                "text": "Unknown",
                "locations": [{"offset": 0, "length": 7}],
            },
        ]
    }

    ents = extract_entities(passage)

    # --- General checks ---
    assert isinstance(ents, list)
    assert len(ents) == 4  # all parsed, even invalid ones are handled gracefully

    # --- Entity 1 (Gene TP53) ---
    g1 = ents[0]
    assert g1["name"] == "TP53"
    assert g1["mention"] == "TP53"
    assert g1["type"] == "Gene"
    assert g1["id"] == "G1"
    assert g1["start"] == 5
    assert g1["end"] == 9  # 5 + 4

    # --- Entity 2 (Chem Aspirin) ---
    c1 = ents[1]
    assert c1["name"] == "Aspirin"
    assert c1["mention"] == "Aspirin"
    assert c1["type"] == "Chem"
    assert c1["id"] == "C1"
    assert c1["start"] == 12
    assert c1["end"] == 19  # 12 + 7

    # --- Entity 3 (malformed, missing location) ---
    d1 = ents[2]
    assert d1["name"] == "Cancer"
    assert d1["start"] is None
    assert d1["end"] is None

    # --- Entity 4 (missing infons) ---
    unk = ents[3]
    assert unk["mention"] == "Unknown"
    assert unk["type"] is None
    assert unk["id"] is None
    assert unk["start"] == 0
    assert unk["end"] == 7


def test_build_publication_metadata():
    meta = build_publication_metadata("123", "Title", "Abstract text.", "2024")
    assert meta == {"pmid": "123", "title": "Title", "abstract": "Abstract text.", "year": "2024"}


def test_process_record_composes_metadata_and_entries():
    record = {
        "id": "999",
        "passages": [
            {"infons": {"type": "title", "year": "2021"}, "text": "Some title", "annotations": []},
            {
                "infons": {"type": "abstract"},
                "text": "Sentence with gene.",
                "annotations": [
                    {
                        "text": "gene",
                        "infons": {"type": "Gene", "identifier": "G1", "name": "gene"},
                        "locations": [{"offset": 15, "length": 4}],
                    }
                ],
            },
        ],
    }
    entries, meta = process_record(record, "999")
    assert meta["pmid"] == "999"
    assert len(entries) == 1
    assert entries[0]["entities"][0]["id"] == "G1"


@patch("data_processing.utils.pubtator.fetch_pubtator_batch")
@patch("data_processing.utils.pubtator.save_cache_pubtator_data")
@patch("data_processing.utils.pubtator.save_publication_metadata")
@patch("data_processing.utils.pubtator.update_processed_pmids")
@patch("data_processing.utils.pubtator.load_cache_pmids", return_value=set())
@patch("time.sleep", return_value=None)
def test_fetch_pubtator_data_integration(
    _mock_sleep: MagicMock,
    _mock_load: MagicMock,
    mock_update: MagicMock,
    mock_save_meta: MagicMock,
    mock_save_cache: MagicMock,
    mock_fetch_batch: MagicMock,
) -> None:
    """Integration test: ensure fetch_pubtator_data orchestrates all steps and caches correctly."""
    mock_fetch_batch.return_value = [
        {
            "pmid": "111",
            "pubYear": 2020,
            "title": "Title A",
            "sentences": [
                {
                    "text": "Sentence A1",
                    "annotations": [
                        {
                            "text": "geneA",
                            "type": "Gene",
                            "normalized": "G1",
                            "offset": 0,
                            "length": 5,
                        },
                    ],
                },
                {
                    "text": "Sentence A2",
                    "annotations": [],
                },
            ],
        },
        {
            "pmid": "222",
            "pubYear": 2019,
            "title": "Title B",
            "sentences": [{"text": "Sentence B", "annotations": []}],
        },
    ]

    fetch_pubtator_data([111, 222])

    mock_fetch_batch.assert_called_once()
    mock_save_cache.assert_called_once()
    mock_save_meta.assert_called_once()
    mock_update.assert_called_once_with(["111", "222"])
