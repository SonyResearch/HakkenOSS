import json
from pathlib import Path

import requests
from ml_utils import DSVUtils
from tqdm import tqdm

from misc.hypotheses_df_reader import HypothesesDFReader

SUCCESS_CODE = 200


def call_api(url, data):
    json_data = json.dumps(data)

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json_data, headers=headers)

    # Check the response
    if response.status_code == SUCCESS_CODE:
        # Request was successful
        result = response.json()
        return result
    # There was an error
    print(f"Error: {response.status_code} {response.text}")
    return None


def main():
    hypotheses_folder = Path(
        "/home/ubuntu/Documents/GitHub/data/hakken_bio/hypotheses_batch_2/prediction_postprocess_output"
    )

    shortest_path_len_url = "http://localhost:8088/test/xkgc/shortest_path_length"

    df = HypothesesDFReader.read_many(
        root_folder=hypotheses_folder,
        read_kwargs={
            "delimiter": "\t",
            "header": 0,
            "use_cols": [
                "Concept pair names (A, B)",
                "Concept pair ocids (A, B)",
                "Relation",
                "Confidence",
            ],
        },
        entity_pair_column="Concept pair ocids (A, B)",
        relation_column="Relation",
        entity_pair_regex=r"(\d+)\s*<=====>?\s*(\d+)",
        score_column="Confidence",
    )

    triples_to_probe = []

    for _idx, row in tqdm(df.iterrows(), total=len(df)):

        triple = [row["subject"], row["relation"], row["object"]]
        triples_to_probe.append(triple)

    data = {
        "triples_to_probe": triples_to_probe,
    }

    output_i_all: dict = call_api(shortest_path_len_url, data)
    if output_i_all is None:
        raise ValueError()

    lengths = [l for key, l in output_i_all["length_dict"].items()]

    df["complexity"] = lengths

    df["hypothesis_type"] = df["hypothesis_type"].str.split("_inference").str[0]

    DSVUtils.write_dsv(df, file_path="batch_2_all_complexity_ocid.tsv", delimiter="\t")

    import matplotlib.pyplot as plt

    counts = df["complexity"].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    counts.plot(kind="bar")

    plt.title("Distribution of Complexity Values")
    plt.xlabel("Complexity")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("complexity_distribution.png")
    plt.close()


if __name__ == "__main__":
    main()
