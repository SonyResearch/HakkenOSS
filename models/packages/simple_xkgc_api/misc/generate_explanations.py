import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from ml_base_structures import Triple
from ml_utils import DSVUtils
from tqdm import tqdm

logging.getLogger().setLevel(logging.WARNING)


SUCCESS_CODE = 200


def call_api(url: str, data: dict):
    json_data = json.dumps(data)

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json_data, headers=headers)

    if response.status_code == SUCCESS_CODE:
        result = response.json()
        logging.info(f"Prediction result: {result}")
        return result
    print(f"Error: {response.status_code} {response.text}")
    return None


def map_explanation(explanation: str, node_mapping: dict[str, str]) -> str:

    triples_list = explanation.split(" <> ")

    explanation_mapped_list = []
    for triple in triples_list:
        subject, relation, object = split_triple_string(triple)
        triple_mapped = (
            f"{node_mapping[subject]} - [{relation}] -> {node_mapping[object]}"
        )

        explanation_mapped_list.append(triple_mapped)

    return " ||| ".join(explanation_mapped_list)


def split_triple_string(triple_str):
    # Remove the outer brackets and split by arrow
    inner_str = triple_str.strip("[]")
    parts = inner_str.split("->")

    # Split the first part to get source ID and relationship type
    source_parts = parts[0].split("-")

    return [
        source_parts[0],  # Source ID
        source_parts[1],  # Relationship type
        parts[1],  # Target ID
    ]


def generate_explanations(
    filename: Path,
    filename_out: Path,
    config: dict[str, Any],
    node_mapping: dict[str, str],
    endpoint: str = "http://localhost:8088/test/xkgc/shortest_path",
) -> None:
    df = DSVUtils.read_dsv(file_path=filename, delimiter="\t", header=0)

    row_list = []

    for _idx, row in tqdm(df.iterrows(), total=len(df)):
        subject_object_ocids = row["Concept pair ocids (A, B)"]
        subject_ocid, object_ocid = re.search(
            r"(\d+)\s*<=====>?\s*(\d+)", subject_object_ocids
        ).groups()

        relation = row["Relation"].upper()

        triple_as_list = [subject_ocid, relation, object_ocid]
        triple = Triple.from_list(triple_as_list)
        data = config.copy()

        data["triples_to_probe"] = [triple_as_list]

        output_i: dict = call_api(endpoint, data)
        explanations_i = output_i["explanations"][str(triple)]

        lengths_i = []
        row_i = {}
        for j, explanation_ij in enumerate(explanations_i):
            row_i[f"Explanation {j+1}"] = map_explanation(
                explanation_ij["data"], node_mapping
            )
            lengths_i.append(explanation_ij["length"])

        assert len(set(lengths_i)) == 1
        row_i["Complexity"] = lengths_i[0]

        row_list.append(row_i)

    df_extra = pd.DataFrame(row_list)
    df_expl = pd.concat([df, df_extra], axis=1)

    df_expl.to_csv(filename_out, sep="\t", index=False, header=True)
