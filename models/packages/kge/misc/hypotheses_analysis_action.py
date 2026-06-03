from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from hakken_ml_toolkit.ml_utils import DSVUtils
from hakken_ml_toolkit.ml_utils.extras import NetworkXUtils
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("graph_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def pairplot(df_plot: pd.DataFrame, filename: Path | str, title: str | None = None):
    sns.pairplot(
        df_plot,
        diag_kind="hist",
        diag_kws={
            "bins": 10,
            "edgecolor": "black",
        },
    )
    if title is not None:
        plt.suptitle(title, y=1.02)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


class HypothesesAnalysisAction(Protocol):
    @staticmethod
    def plot_correlation(
        df: pd.DataFrame,
        filename: Path,
        columns: list[str] | None = None,
    ) -> None:
        if columns is None:
            columns = ["score", "len_shortest_path"]
        sns.heatmap(df[columns].corr(), annot=True, cmap="coolwarm")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        filename_no_ext = filename.with_suffix("")

        df_sample = df[columns].sample(1000)
        pairplot(df_sample, filename=str(filename_no_ext) + "_pairplot.png")

        score_ranges = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (0.95, 1.0)]

        for i, (min_score, max_score) in enumerate(score_ranges):
            mask = (df["score"] >= min_score) & (df["score"] < max_score)
            df_filtered = df[mask][columns]

            if len(df_filtered) > 1000:
                df_filtered = df_filtered.sample(1000)

            pairplot(
                df_filtered,
                filename=str(filename_no_ext) + f"_pairplot_{i}.png",
                title=f"Score Range: {min_score:.2f} - {max_score:.2f}\n(n={len(df_filtered)})",
            )

    @staticmethod
    def plot_len_shortest_path(
        df: pd.DataFrame,
        filename: Path,
        shortest_path_column: str = "len_shortest_path",
        hypothesis_type_column: str = "hypothesis_type",
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

        sns.histplot(
            data=df, x=shortest_path_column, ax=ax1, bins=30, stat="count"
        )  # Changed to count
        ax1.set_title("Overall Distribution of Shortest Path Lengths")
        ax1.set_xlabel("Shortest Path Length")
        ax1.set_ylabel("Count")

        sns.histplot(
            data=df,
            x=shortest_path_column,
            hue=hypothesis_type_column,
            ax=ax2,
            bins=30,
            stat="count",  # Changed to count
            multiple="stack",  # Changed to stack for better count visualization
        )
        ax2.set_title("Distribution of Shortest Path Lengths by Hypothesis Type")
        ax2.set_xlabel("Shortest Path Length")
        ax2.set_ylabel("Count")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches="tight")

        plt.close()

    @staticmethod
    def read_knowledge_graph(
        filepath_csv: Path, filepath_nx: Path, is_directed: bool = False
    ) -> nx.Graph:
        if filepath_nx.exists():
            logger.info("Loading existing graph from pickle file")
            with open(filepath_nx, "rb") as f:
                return cast(nx.Graph, pickle.load(f))

        logger.info("Creating new graph from edges data")
        df = DSVUtils.read_dsv(
            filepath_csv,
            delimiter="\t",
            header=0,
            dtype={"ocid_subject": str, "ocid_object": str, "ocid_relation": str},
        )
        edges = df[["ocid_subject", "ocid_object"]].drop_duplicates()
        graph: nx.DiGraph = nx.from_pandas_edgelist(
            edges, source="ocid_subject", target="ocid_object", create_using=nx.DiGraph
        )

        if not is_directed:
            graph = graph.to_undirected()

        logger.info(
            f"Graph created with {graph.number_of_nodes()} nodes and"
            f" {graph.number_of_edges()} edges"
        )
        with open(filepath_nx, "wb") as f:
            pickle.dump(graph, f)

        return graph

    def compute_spath(
        self: pd.DataFrame, graph: nx.DiGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        shorest_path_lenghts = []
        paths = []
        for _i, (_idx, hypothesis) in tqdm(
            enumerate(self.iterrows()),
            total=len(self),
            desc="Computing shortest paths",
        ):
            source = hypothesis["subject"]
            target = hypothesis["object"]
            shortest_path = NetworkXUtils.shortest_path(
                graph, source=source, target=target
            )
            len_shortest_path = len(shortest_path)

            if len_shortest_path == 0:
                shorest_path_lenghts.append(-1)
            else:
                shorest_path_lenghts.append(len_shortest_path)

            if len_shortest_path > 2:
                paths.append(";".join(shortest_path[1:-1]))  # type: ignore[arg-type]
            else:
                paths.append("")

        path_lengths = np.array(shorest_path_lenghts)
        paths_arr = np.array(paths)

        return path_lengths, paths_arr

    @staticmethod
    def read_many_raw_hypotheses_df(
        root_folder: Path,
        read_kwargs: dict[str, Any] | None = None,
        entity_pair_column: str | int = 1,
        entity_pair_regex: str = r"(\d+)\) <=====> (\d+)\)",
        score_column: str | int = 3,
    ) -> pd.DataFrame:
        df_list = []
        for root, _dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(".csv"):
                    filename = Path(os.path.join(root, file))

                    df = HypothesesAnalysisAction.read_raw_hypotheses_df(
                        filename=filename,
                        read_kwargs=read_kwargs,
                        entity_pair_column=entity_pair_column,
                        entity_pair_regex=entity_pair_regex,
                        score_column=score_column,
                    )
                    df["hypothesis_type"] = filename.name.replace(
                        "_inference_output_post.csv", ""
                    )

                    df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def read_raw_hypotheses_df(
        filename: Path,
        read_kwargs: dict[str, Any] | None = None,
        entity_pair_column: str | int = 1,
        entity_pair_regex: str = r"(\d+)\) <=====> (\d+)\)",
        score_column: str | int = 3,
    ) -> pd.DataFrame:
        if read_kwargs is None:
            read_kwargs = {"delimiter": "\t"}

        logger.info(f"Loading {filename}...")
        df_raw = DSVUtils.read_dsv(filename, **read_kwargs)
        logger.info(f"Loaded raw data: {len(df_raw)} rows {df_raw.columns}")

        df_hypotheses = df_raw[entity_pair_column].str.extract(entity_pair_regex)
        df_hypotheses.rename({0: "subject", 1: "object"}, axis=1, inplace=True)
        df_hypotheses["score"] = df_raw[score_column].astype(float)
        df_hypotheses = df_hypotheses.dropna()
        logger.info(f"Processed scores data: {len(df_hypotheses)} rows after cleaning")
        return df_hypotheses
