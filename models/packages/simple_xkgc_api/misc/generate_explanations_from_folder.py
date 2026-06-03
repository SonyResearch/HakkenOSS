import argparse
import logging
import os
from pathlib import Path

from ml_utils import DSVUtils
from tqdm import tqdm

logging.getLogger().setLevel(logging.WARNING)


from misc.generate_explanations import generate_explanations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypotheses_folder",
        type=Path,
        required=True,
        help="Path to hypotheses folder",
    )
    # parser.add_argument(
    #     "--nodes_file", type=str, required=True, help="Path to nodes CSV file"
    # )
    # parser.add_argument(
    #     "--shortest_path_url",
    #     type=str,
    #     default="http://localhost:8088/test/xkgc/shortest_path",
    #     help="URL for shortest path API",
    # )
    # parser.add_argument(
    #     "--num_explanations",
    #     type=int,
    #     default=5,
    #     help="Number of explanations to generate",
    # )
    return parser.parse_args()


def main(hypotheses_folder: Path):

    df_nodes_file_path = Path(
        "/home/ubuntu/Documents/GitHub/data/hakken_bio/v2/nodes.csv"
    )

    df_nodes = DSVUtils.read_dsv(
        df_nodes_file_path,
        delimiter="\t",
        header=0,
        dtype={"ocid_node": str, "ocid_domain": str},
    )

    node_mapping = dict(zip(df_nodes["ocid_node"], df_nodes["node"]))

    total_dirs = sum(1 for _ in os.walk(hypotheses_folder))

    for root, _dirs, files in tqdm(
        os.walk(hypotheses_folder), total=total_dirs, desc="Processing directories"
    ):
        for file in tqdm(
            files, desc=f"Processing files in {os.path.basename(root)}", leave=False
        ):
            if file.endswith(".csv"):
                filename = Path(os.path.join(root, file))
                filename_out = filename.with_suffix(".tsv").with_stem(
                    filename.stem + "_explanations"
                )

                generate_explanations(
                    filename=filename,
                    filename_out=filename_out,
                    config={
                        "num_explanations": 5,
                    },
                    node_mapping=node_mapping,
                    endpoint="http://localhost:8088/test/xkgc/shortest_path",
                )


if __name__ == "__main__":
    args = parse_args()

    main(hypotheses_folder=args.hypotheses_folder)
