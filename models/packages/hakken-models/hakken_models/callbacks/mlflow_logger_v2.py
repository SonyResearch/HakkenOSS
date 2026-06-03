import tempfile
from pathlib import Path

import mlflow
import yaml
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from torch import Tensor


class MLFlowLoggerV2(MLFlowLogger):
    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, local_path, s, tag in checkpoints:
            metadata = {
                # Ensure .item() is called to store Tensor contents
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(local_path).name,
                "Checkpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    # ensure it does not break if `Checkpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            aliases = (
                ["latest", "best"]
                if local_path == checkpoint_callback.best_model_path
                else ["latest"]
            )
            if tag not in aliases:
                aliases.append(tag)

            # Artifact path on mlflow
            local_path_stem = Path(local_path).stem

            mlflow.log_artifact(local_path=local_path, artifact_path="checkpoints")

            with tempfile.TemporaryDirectory() as tmp_dir:
                metadata_path = f"{tmp_dir}/{local_path_stem}-metadata.yaml"

                # Log the metadata
                with open(metadata_path, "w") as tmp_file_metadata:
                    yaml.dump(metadata, tmp_file_metadata, default_flow_style=False)

                aliases_path = f"{tmp_dir}/{local_path_stem}-aliases.txt"
                # Log the aliases
                with open(aliases_path, "w") as tmp_file_aliases:
                    tmp_file_aliases.write(str(aliases))

                mlflow.log_artifact(metadata_path, artifact_path="checkpoints")
                mlflow.log_artifact(aliases_path, artifact_path="checkpoints")

            self._logged_model_time[local_path] = t
