"""Benchmark degree calculation speed of NetworkX-based (i.e. file-based) graph implementation."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import psutil

from filtering.core.entities.kg import EdgeDirection
from filtering.impl.kg.networkx_kg import NetworkXKnowledgeGraph, NetworkXKnowledgeGraphConfig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph_dir", required=True, type=str)
    parser.add_argument("--test_data_dir", required=True, type=str)
    parser.add_argument("--write_dir", required=True, type=str)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"

    test_data_dir = Path(args.test_data_dir)

    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    write_path = write_dir / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"

    result = {}
    for test_file in test_data_dir.glob("*.txt"):
        test_name = test_file.stem
        print(f"*** {test_name} ***")

        test_result = {}

        tic = time.perf_counter()
        config = NetworkXKnowledgeGraphConfig(
            nodes_path=nodes_path,
            edges_path=edges_path,
            node_id_column_name="id",
            node_ocid_column_name="ocid",
            edge_start_column_name="start_id",
            edge_end_column_name="end_id",
            edge_time_column_name="timestamp",
            edge_is_time_column_timestamp=True,
            edge_num_occurrences_column_name="number_of_occurrences",
        )
        graph = NetworkXKnowledgeGraph(config)
        toc = time.perf_counter()
        test_result["init"] = toc - tic
        print(f"- init: {test_result['init']} s")
        print(f"- memory used: {psutil.Process().memory_info().rss / (1024 * 1024)} MB")

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
