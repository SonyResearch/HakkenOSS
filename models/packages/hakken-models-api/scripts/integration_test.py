#!/usr/bin/env python3
"""Integration test script for hakken-models-api.

Tests a running API instance by exercising its endpoints.
Expects the API to be running (e.g. via `make serve`).

Usage:
    uv run python scripts/integration_test.py
    uv run python scripts/integration_test.py --base-url http://localhost:8088 --model segal
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

HTTP_OK = 200
NUM_SAMPLE_FACTS = 3


def request(
    url: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    """Perform HTTP request and return (status_code, body)."""
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
    else:
        req = urllib.request.Request(url, method=method)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            return resp.status, parsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return e.code, parsed
    except urllib.error.URLError as e:
        return -1, str(e.reason)


def run_test(name: str, ok: bool, detail: str = "") -> bool:
    """Print test result and return success."""
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok and detail:
        print(f"         {detail}")
    return ok


def test_segal(base_url: str, path_prefix: str = "") -> bool:
    """Test SeGAL endpoints."""
    base = base_url.rstrip("/")
    prefix = f"{base}{path_prefix}/segal"
    all_ok = True

    # GET /segal/info
    status, body = request(f"{prefix}/info")
    ok = status == HTTP_OK and isinstance(body, dict)
    if ok:
        ok = all(k in body for k in ("num_entities", "num_relations", "has_embeddings"))
    all_ok &= run_test(
        "GET /segal/info",
        ok,
        f"status={status}" if not ok else "",
    )

    # POST /segal/sample-facts
    status, body = request(
        f"{prefix}/sample-facts",
        method="POST",
        data={"num_samples": NUM_SAMPLE_FACTS, "splits": ["train"]},
    )
    ok = status == HTTP_OK and isinstance(body, dict) and "facts_list" in body
    if ok:
        ok = len(body["facts_list"]) == NUM_SAMPLE_FACTS
    all_ok &= run_test(
        "POST /segal/sample-facts",
        ok,
        f"status={status}" if not ok else "",
    )

    # Use sampled facts for downstream tests
    facts_list: list[tuple[str, str, str]] = []
    if ok and isinstance(body, dict):
        facts_list = body.get("facts_list", [])

    # POST /segal/fact-indexes
    status, body = request(
        f"{prefix}/fact-indexes",
        method="POST",
        data={"facts_list": facts_list[:1] if facts_list else []},
    )
    ok = status == HTTP_OK and isinstance(body, dict) and "fact_index_list" in body
    all_ok &= run_test(
        "POST /segal/fact-indexes",
        ok,
        f"status={status}" if not ok else "",
    )

    # POST /segal/score (only if we have facts; score endpoint expects {"request": ...} wrapper)
    if facts_list:
        status, body = request(
            f"{prefix}/score",
            method="POST",
            data={"request": {"facts_list": facts_list[:1]}},
        )
        ok = status == HTTP_OK and isinstance(body, dict)
        if ok:
            ok = "scores_list" in body and "normalized_scores_list" in body
        all_ok &= run_test(
            "POST /segal/score",
            ok,
            f"status={status}" if not ok else "",
        )
    else:
        run_test("POST /segal/score", False, "SKIP: no facts from sample-facts")

    # POST /segal/score-text
    score_text_data: dict[str, Any] = {
        "target_facts": [
            {
                "subject": {"name": "PARACETAMOL", "domain": "DRUG"},
                "relation": {"name": "TREATS"},
                "object": {"name": "HEADACHE", "domain": "DISEASE"},
            }
        ],
        "context_facts": [
            {
                "subject": {"name": "ASPIRIN", "domain": "DRUG"},
                "relation": {"name": "TREATS"},
                "object": {"name": "PAIN", "domain": "SYMPTOM"},
                "timestamp": 2024.0,
            },
            {
                "subject": {"name": "IBUPROFEN", "domain": "DRUG"},
                "relation": {"name": "TREATS"},
                "object": {"name": "HEADACHE", "domain": "DISEASE"},
                "timestamp": 2024.0,
            },
        ],
    }
    status, body = request(
        f"{prefix}/score-text",
        method="POST",
        data=score_text_data,
    )
    ok = status == HTTP_OK and isinstance(body, dict)
    if ok:
        ok = (
            "scores_list" in body
            and "normalized_scores_list" in body
            and len(body["scores_list"]) == 1
        )
    all_ok &= run_test(
        "POST /segal/score-text",
        ok,
        f"status={status}, body={body}" if not ok else "",
    )

    # POST /segal/score-text (no context)
    score_text_no_ctx: dict[str, Any] = {
        "target_facts": [
            {
                "subject": {"name": "ASPIRIN", "domain": "DRUG"},
                "relation": {"name": "TREATS"},
                "object": {"name": "PAIN", "domain": "SYMPTOM"},
            }
        ],
    }
    status, body = request(
        f"{prefix}/score-text",
        method="POST",
        data=score_text_no_ctx,
    )
    ok = status == HTTP_OK and isinstance(body, dict)
    if ok:
        ok = "scores_list" in body and len(body["scores_list"]) == 1
    all_ok &= run_test(
        "POST /segal/score-text (no context)",
        ok,
        f"status={status}, body={body}" if not ok else "",
    )

    return all_ok


def test_thiger(base_url: str, path_prefix: str = "") -> bool:
    """Test THiGER endpoints."""
    base = base_url.rstrip("/")
    prefix = f"{base}{path_prefix}/thiger"
    all_ok = True

    # POST /thiger/sample-facts
    status, body = request(
        f"{prefix}/sample-facts",
        method="POST",
        data={"num_samples": NUM_SAMPLE_FACTS, "splits": ["train"]},
    )
    ok = status == HTTP_OK and isinstance(body, dict) and "facts_list" in body
    if ok:
        ok = len(body["facts_list"]) == NUM_SAMPLE_FACTS
    all_ok &= run_test(
        "POST /thiger/sample-facts",
        ok,
        f"status={status}" if not ok else "",
    )

    facts_list: list[tuple[str, str, str]] = []
    if ok and isinstance(body, dict):
        facts_list = body.get("facts_list", [])

    # POST /thiger/fact-indexes
    status, body = request(
        f"{prefix}/fact-indexes",
        method="POST",
        data={"facts_list": facts_list[:1] if facts_list else []},
    )
    ok = status == HTTP_OK and isinstance(body, dict) and "fact_index_list" in body
    all_ok &= run_test(
        "POST /thiger/fact-indexes",
        ok,
        f"status={status}" if not ok else "",
    )

    # POST /thiger/score (score endpoint expects {"request": ...} wrapper)
    if facts_list:
        status, body = request(
            f"{prefix}/score",
            method="POST",
            data={"request": {"facts_list": facts_list[:1]}},
        )
        ok = status == HTTP_OK and isinstance(body, dict)
        if ok:
            ok = "scores_list" in body and "normalized_scores_list" in body
        all_ok &= run_test(
            "POST /thiger/score",
            ok,
            f"status={status}" if not ok else "",
        )
    else:
        run_test("POST /thiger/score", False, "SKIP: no facts from sample-facts")

    # POST /thiger/predict (requires entity IDs from dataset; expects {"request": ...} wrapper)
    if facts_list:
        s, r, o = facts_list[0]
        status, body = request(
            f"{prefix}/predict",
            method="POST",
            data={
                "request": {
                    "subject_id_list": [s],
                    "object_id_list": [o],
                }
            },
        )
        ok = status == HTTP_OK and isinstance(body, dict)
        if ok:
            ok = "relations_ids" in body and "relations_scores" in body
        all_ok &= run_test(
            "POST /thiger/predict",
            ok,
            f"status={status}" if not ok else "",
        )
    else:
        run_test("POST /thiger/predict", False, "SKIP: no facts from sample-facts")

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integration test for hakken-models-api (expects running server)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8088",
        help="Base URL of the running API (default: http://localhost:8088)",
    )
    parser.add_argument(
        "--path-prefix",
        default="/test",
        help="Path prefix for model routes (default: /test, used by spaice-inference-api)",
    )
    parser.add_argument(
        "--model",
        choices=["segal", "thiger"],
        default="segal",
        help="Model type to test (default: segal)",
    )
    args = parser.parse_args()

    full_url = f"{args.base_url.rstrip('/')}{args.path_prefix}"
    print(f"Testing {args.model.upper()} API at {full_url}\n")

    if args.model == "segal":
        ok = test_segal(args.base_url, args.path_prefix)
    else:
        ok = test_thiger(args.base_url, args.path_prefix)

    print()
    if ok:
        print("All tests passed.")
        return 0
    print("Some tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
