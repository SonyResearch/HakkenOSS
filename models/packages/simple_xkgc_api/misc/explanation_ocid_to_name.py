from pathlib import Path

from ml_utils import DSVUtils


def split_triple_string(triple_str):
    inner_str = triple_str.strip("[]")
    parts = inner_str.split("->")

    # Split the first part to get source ID and relationship type
    source_parts = parts[0].split("-")

    return [
        source_parts[0],  # Source ID
        source_parts[1],  # Relationship type
        parts[1],  # Target ID
    ]


def main():

    df_nodes_file_path = Path(
        "/home/ubuntu/Documents/GitHub/data/hakken_bio/v2/nodes.csv"
    )

    df_complexity_file_path = Path(
        "/home/ubuntu/Documents/GitHub/project_spaice_ds/packages/pip/simple_xkgc_api/batch_2_complexity_ocid.tsv"
    )

    df_explanation_file_path = Path(
        "/home/ubuntu/Documents/GitHub/project_spaice_ds/packages/pip/simple_xkgc_api/batch_2_explanations_ocid.tsv"
    )

    df_nodes = DSVUtils.read_dsv(
        df_nodes_file_path,
        delimiter="\t",
        header=0,
        dtype={"ocid_node": str, "ocid_domain": str},
    )

    node_mapping = dict(zip(df_nodes["ocid_node"], df_nodes["node"]))

    df_complexity = DSVUtils.read_dsv(
        df_complexity_file_path,
        delimiter="\t",
        header=0,
        dtype={"subject": str, "object": str},
    )

    df_complexity["subject_str"] = df_complexity["subject"].map(node_mapping)

    df_complexity["object_str"] = df_complexity["object"].map(node_mapping)

    DSVUtils.write_dsv(
        df_complexity, file_path="batch_2_complexity.tsv", delimiter="\t"
    )

    df_explanations = DSVUtils.read_dsv(
        df_explanation_file_path,
        delimiter="\t",
        header=0,
        dtype={"subject": str, "object": str},
    )

    df_explanations["subject_str"] = df_explanations["subject"].map(node_mapping)

    df_explanations["object_str"] = df_explanations["object"].map(node_mapping)

    explanations_str_list = []

    for _idx, row in df_explanations.iterrows():

        print(row["explanation"])
        explanation_mapped = map_explanation(row["explanation"], node_mapping)
        explanations_str_list.append(explanation_mapped)

    df_explanations["explanation_str"] = explanations_str_list

    DSVUtils.write_dsv(
        df_explanations, file_path="batch_2_explanations.tsv", delimiter="\t"
    )


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


if __name__ == "__main__":
    main()
