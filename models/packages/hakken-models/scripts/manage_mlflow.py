"""
MLflow helpers: export run params or metrics to YAML.

Run from the hakken-models package root, e.g.:

    uv run python scripts/manage_mlflow.py export-params <run_id> -o params.yaml
    uv run python scripts/manage_mlflow.py export-metrics <run_id> -o metrics.yaml
"""

from __future__ import annotations

import pathlib  # noqa: TC003

import mlflow
import typer
from loguru import logger
from omegaconf import OmegaConf

from hakken_models.core.utils.data import unflatten_dict

app = typer.Typer(help="MLflow utilities for hakken-models")


def _set_tracking_uri(tracking_uri: str | None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def _metrics_dict_from_run(run: mlflow.entities.Run) -> dict[str, float]:
    """Latest value per metric key from an MLflow Run."""
    raw = run.data.metrics
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected run.data.metrics type: {type(raw)}")
    return {k: float(v) for k, v in raw.items()}


def _write_yaml(payload: dict, output: pathlib.Path | None) -> None:
    cfg = OmegaConf.create(payload)
    text = OmegaConf.to_yaml(cfg)
    if output is None or str(output) == "-":
        typer.echo(text.rstrip("\n"))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Wrote {output}")


@app.command("export-params")
def export_params(
    run_id: str = typer.Argument(..., help="MLflow run ID"),
    output: pathlib.Path | None = typer.Option(  # noqa: B008
        None,
        "-o",
        "--output",
        help="YAML file path; omit or '-' for stdout",
    ),
    mlflow_tracking_uri: str | None = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="Tracking URI (optional if MLFLOW_TRACKING_URI is set).",
    ),
    flat: bool = typer.Option(
        False,
        "--flat",
        help="Export flat string params as logged (no '/' unflatten).",
    ),
    strip_checkpoints: bool = typer.Option(
        True,
        "--strip-checkpoints/--keep-checkpoints",
        help="When nested, drop last_checkpoint_path and checkpoint_dir.",
    ),
) -> None:
    """Export a run's logged params to YAML (nested by default, '/'-separated keys)."""
    _set_tracking_uri(mlflow_tracking_uri)
    run = mlflow.get_run(run_id=run_id)
    flat_params = dict(run.data.params)
    if not flat_params:
        raise typer.BadParameter("Run has no params.")

    if flat:
        payload: dict = flat_params
    else:
        if strip_checkpoints:
            skip = {"last_checkpoint_path", "checkpoint_dir"}
            flat_params = {k: v for k, v in flat_params.items() if k not in skip}
        payload = unflatten_dict(flat_params, sep="/")

    _write_yaml(payload, output)


@app.command("export-metrics")
def export_metrics(
    run_id: str = typer.Argument(..., help="MLflow run ID"),
    output: pathlib.Path | None = typer.Option(  # noqa: B008
        None,
        "-o",
        "--output",
        help="YAML file path; omit or '-' for stdout",
    ),
    mlflow_tracking_uri: str | None = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="Tracking URI (optional if MLFLOW_TRACKING_URI is set).",
    ),
) -> None:
    """Export each metric's latest value for the run to YAML."""
    _set_tracking_uri(mlflow_tracking_uri)
    run = mlflow.get_run(run_id=run_id)
    metrics = _metrics_dict_from_run(run)
    if not metrics:
        logger.warning("Run has no metrics (empty export).")
    _write_yaml(metrics, output)


if __name__ == "__main__":
    app()
