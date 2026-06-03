from typing import Annotated

import polars as pl
from evidently import Report
from evidently.presets import DataSummaryPreset
from zenml import step
from zenml.types import HTMLString


@step
def generate_data_quality_report_step(
    df: pl.DataFrame, num_samples: int | None = None, seed: int = 42
) -> tuple[
    Annotated[str, "{split_name}_report_json"], Annotated[HTMLString, "{split_name}_report_html"]
]:
    if df.height == 0:
        return "", HTMLString("")
    # Convert to pandas for Evidently compatibility
    n_samples = min(num_samples, df.height) if num_samples is not None else df.height
    if n_samples < df.height:
        data_pd = df.sample(n=n_samples, seed=seed).to_pandas()
    else:
        data_pd = df.to_pandas()

    # Quality reports
    report = Report(metrics=[DataSummaryPreset()])
    eval = report.run(data_pd)
    return eval.json(), HTMLString(eval.get_html_str(as_iframe=False))
