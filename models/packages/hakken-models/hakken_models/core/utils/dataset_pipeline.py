from typing import cast

import numpy as np
import polars as pl
import torch
from zenml.models import PipelineRunResponse


def load_facts_np(run: PipelineRunResponse, split_name: str) -> np.ndarray:
    return cast(np.ndarray, run.steps[f"build_{split_name}_tensor_step"].output.load())


def load_facts_tensor(
    run: PipelineRunResponse, split_name: str, device: str | None = None
) -> torch.Tensor:
    facts_np = run.steps[f"build_{split_name}_tensor_step"].output.load()
    return torch.tensor(facts_np, dtype=torch.long, device=device)


def load_map_df(run: PipelineRunResponse, name: str) -> pl.DataFrame:
    return cast(pl.DataFrame, run.steps[f"build_{name}_mapping"].output.load())
