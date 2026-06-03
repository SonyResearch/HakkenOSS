from datetime import datetime, timedelta
from pathlib import Path

import pytz
from ml_utils import DSVUtils
from omegaconf import OmegaConf
from pydantic import field_validator
from pydantic_settings import BaseSettings
from file_manager import S3Manager
from tqdm import tqdm

from misc.generate_explanations import generate_explanations


class DownloadFilesSettings(BaseSettings):
    bucket_name: str
    hypotheses_path: Path
    local_folder: Path
    df_nodes_file_path: Path
    last_run: datetime  # In ISO format YYYY-MM-DDTHH:MM:SS+HH:MM

    @field_validator("last_run", mode="before")
    def parse_last_run(cls, v: str | datetime) -> datetime:
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(v)


def parse_args() -> DownloadFilesSettings:

    base_conf = {
        "bucket_name": "sai-spaice-ds",
        "hypotheses_path": "ds_project_files/hypotheses_selection_pipeline/batch_3/hypotheses_folder",
        "local_folder": "/home/ubuntu/Documents/GitHub/project_spaice_ds/packages/pip/simple_xkgc_api/data/batch_test",
        "df_nodes_file_path": "/home/ubuntu/Documents/GitHub/data/hakken_bio/v2/nodes.csv",
        "last_run": (datetime.now(pytz.UTC) - timedelta(days=100)).isoformat(),
    }

    cli_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(OmegaConf.create(base_conf), cli_conf)

    return DownloadFilesSettings(**OmegaConf.to_container(config))


def main(cfg: DownloadFilesSettings) -> None:

    df_nodes = DSVUtils.read_dsv(
        cfg.df_nodes_file_path,
        delimiter="\t",
        header=0,
        dtype={"ocid_node": str, "ocid_domain": str},
    )

    node_mapping = dict(zip(df_nodes["ocid_node"], df_nodes["node"]))

    client = S3Manager()

    df_hypotheses_all = client.list_objects(
        bucket_name=cfg.bucket_name, prefix=cfg.hypotheses_path
    )

    df_hypotheses = df_hypotheses_all[df_hypotheses_all.last_modified > cfg.last_run]

    filename_list = []
    for _idx, row in tqdm(
        df_hypotheses.iterrows(), total=len(df_hypotheses), desc="Processing rows"
    ):

        key = Path(row["key"])
        filename = client.download_file(
            bucket_name=cfg.bucket_name, key=key, local_folder=cfg.local_folder
        )
        if filename is not None:
            filename_list.append(filename)

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

    return filename_list


if __name__ == "__main__":

    cfg = parse_args()
    filename_list = main(cfg)

    for filename in filename_list:
        print(filename)
