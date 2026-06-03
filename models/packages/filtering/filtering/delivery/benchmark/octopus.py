"""Benchmark degree calculation speed of Neo4j-based graph implementation."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from filtering.core.entities.kg import EdgeDirection
from filtering.impl.kg.neo4j_kg import Neo4jKnowledgeGraph, Neo4jKnowledgeGraphConfig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test_data_dir", required=True, type=str)
    parser.add_argument("--write_dir", required=True, type=str)
    parser.add_argument("--neo4j_username", default=None, type=str)
    parser.add_argument("--neo4j_password", default=None, type=str)
    parser.add_argument("--neo4j_host", default=None, type=str)
    parser.add_argument("--neo4j_port", default=None, type=int)
    args = parser.parse_args()

    test_data_dir = Path(args.test_data_dir)

    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    write_path = write_dir / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"

    load_dotenv()

    neo4j_username = args.neo4j_username or os.getenv("NEO4J_USERNAME")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    neo4j_host = args.neo4j_host or os.getenv("NEO4J_HOST")
    neo4j_port = args.neo4j_port or os.getenv("NEO4J_PORT")

    assert neo4j_username is not None, (
        "Username for Neo4j should be given either by argument or environment variable."
    )
    assert neo4j_password is not None, (
        "Password for Neo4j should be given either by argument or environment variable."
    )

    neo4j_auth = {"username": neo4j_username, "password": neo4j_password}
    if neo4j_host is not None:
        neo4j_auth["host"] = neo4j_host
    if neo4j_port is not None:
        neo4j_auth["port"] = int(neo4j_port)

    result = {}
    for test_file in test_data_dir.glob("*.txt"):
        test_name = test_file.stem
        print(f"*** {test_name} ***")

        test_result = {}

        tic = time.perf_counter()
        config = Neo4jKnowledgeGraphConfig(**neo4j_auth)
        graph = Neo4jKnowledgeGraph(config)
        toc = time.perf_counter()
        test_result["init"] = toc - tic
        print(f"- init: {test_result['init']} s")

        node_ocids = []
        with open(test_file) as f:
            for line in f:
                node_ocids.append(line.strip())

        for direction in [EdgeDirection.IN, EdgeDirection.OUT]:
            tic = time.perf_counter()
            graph.get_degrees(node_ids=node_ocids, direction=direction)
            toc = time.perf_counter()
            test_result[direction] = toc - tic
            print(f"- {direction}: {test_result[direction]} s")

        result[test_name] = test_result

    with open(write_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
