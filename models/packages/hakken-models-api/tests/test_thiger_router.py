# test_thiger_endpoints.py

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8088/test")


def _is_api_available() -> bool:
    """Return True if the API server responds, False otherwise."""
    try:
        with httpx.Client(timeout=2.0) as client:
            client.get(f"{BASE_URL}/openapi.json")
        return True
    except Exception:
        return False


api_available: bool = _is_api_available()

pytestmark = pytest.mark.skipif(
    not api_available,
    reason="API server is not running – skipping endpoint tests",
)


def _post_json(path: str, payload: dict[str, Any]) -> httpx.Response:
    with httpx.Client(timeout=20.0) as client:
        return client.post(f"{BASE_URL}{path}", json=payload)


# -------------------------------------------------------------------------
# Shared fixture: get REAL facts from /thiger/sample-facts
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sampled_facts() -> list[tuple[str, str, str]]:
    """
    Call /thiger/sample-facts and return a list of triples (s, r, o).
    If unavailable/empty, skip dependent tests.
    """
    payload: dict[str, Any] = {
        "splits": ["train"],  # adjust if using val/test only
        "num_samples": 3,
    }

    response = _post_json("/thiger/sample-facts", payload)

    if response.status_code != 200:
        pytest.skip(f"sample-facts unavailable: status={response.status_code}")

    data: dict[str, Any] = response.json()

    if "facts" not in data or not data["facts"]:
        pytest.skip("sample-facts returned no facts")

    facts: list[tuple[str, str, str]] = data["facts"]

    # Make sure triples have correct structure
    assert all(len(f) == 3 for f in facts)

    return facts


# -------------------------------------------------------------------------
# /thiger/predict
# -------------------------------------------------------------------------


def test_thiger_predict(sampled_facts: list[tuple[str, str, str]]) -> None:
    # use first fact
    s, _, o = sampled_facts[0]

    payload: dict[str, Any] = {
        "request": {
            "subject_id_list": [s],
            "object_id_list": [o],
            "inference_config": {},
        }
    }

    response = _post_json("/thiger/predict", payload)
    assert response.status_code == 200

    data = response.json()
    assert "relations_ids" in data
    assert "relations_probs" in data
    assert "relations_scores" in data


# -------------------------------------------------------------------------
# /thiger/score
# -------------------------------------------------------------------------


def test_thiger_score(sampled_facts: list[tuple[str, str, str]]) -> None:
    payload: dict[str, Any] = {
        "request": {
            "facts_list": sampled_facts,  # directly use sampled triples
            "inference_config": {},
        }
    }

    response = _post_json("/thiger/score", payload)
    assert response.status_code == 200

    data = response.json()
    assert "scores_list" in data
    assert "normalized_scores_list" in data


# -------------------------------------------------------------------------
# /thiger/entity-pair-indexes
# -------------------------------------------------------------------------


def test_thiger_entity_pair_indexes(sampled_facts: list[tuple[str, str, str]]) -> None:
    subjects: list[str] = [s for s, _, _ in sampled_facts]
    objects: list[str] = [o for _, _, o in sampled_facts]

    payload: dict[str, Any] = {
        "subject_id_list": subjects,
        "object_id_list": objects,
        "inference_config": {},
    }

    response = _post_json("/thiger/entity-pair-indexes", payload)
    assert response.status_code == 200

    data = response.json()
    assert "subject_index_list" in data
    assert "object_index_list" in data
    assert "entity_pairs" in data


# -------------------------------------------------------------------------
# /thiger/fact-indexes
# -------------------------------------------------------------------------


def test_thiger_fact_indexes(sampled_facts: list[tuple[str, str, str]]) -> None:
    payload: dict[str, Any] = {
        "facts_list": sampled_facts,
        "inference_config": {},
    }

    response = _post_json("/thiger/fact-indexes", payload)
    assert response.status_code == 200

    data = response.json()
    assert "fact_index_list" in data


# -------------------------------------------------------------------------
# /thiger/sample-facts itself
# -------------------------------------------------------------------------


def test_thiger_sample_facts_raw(sampled_facts: list[tuple[str, str, str]]) -> None:
    # The fixture already validated structure; just assert type here
    assert isinstance(sampled_facts, list)
    assert all(isinstance(f, (list, tuple)) for f in sampled_facts)
