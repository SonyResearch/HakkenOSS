import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from hakken_models.results.entities import LegendConfig, SaveFigureConfig
from hakken_models.results.plots.legend import get_legend, save_legend
from hakken_models.results.plots.save_figure import save_figure


def plot_radar_chart(
    df: pd.DataFrame,
    model_col: str = "model",
    normalize: bool = True,
    title: str = "Model comparison (normalized radar chart)",
    remove_legend: bool = False,
    legend_config: LegendConfig | None = None,
    save_config: SaveFigureConfig | None = None,
    save_legend_config: SaveFigureConfig | None = None,
    colors: list | None = None,
    ax: Axes | None = None,
) -> tuple[plt.Figure, Axes]:
    """
    TODO: Add docstring
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    metric_cols = [c for c in df.columns if c != model_col]
    if not metric_cols:
        raise ValueError("No metric columns found.")

    # Ensure numeric metrics
    metrics_df = df[metric_cols].apply(pd.to_numeric, errors="coerce")

    # Normalize metrics to [0, 1] per column
    if normalize:
        col_min = metrics_df.min(axis=0)
        col_max = metrics_df.max(axis=0)
        denom = (col_max - col_min).replace(0, 1)  # avoid /0 if constant metric
        metrics_norm = (metrics_df - col_min) / denom
    else:
        metrics_norm = metrics_df

    models = df[model_col].astype(str).tolist()
    labels = metric_cols
    n_metrics = len(labels)

    # Radar angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    # Plot each model row
    default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, model in enumerate(models):
        values = metrics_norm.iloc[i].tolist()
        values += values[:1]  # close loop
        if colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = default_color_cycle[i % len(default_color_cycle)]

        ax.plot(angles, values, linewidth=2.2, label=model, color=color)
        ax.fill(angles, values, color=color, alpha=0.10)

    # Axis formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", pad=24)

    ax.set_title(title, fontsize=18, pad=20)
    legend = get_legend(ax, legend_config)

    if save_legend_config is not None and legend is not None:
        save_legend(ax, save_legend_config, legend_config)

    if remove_legend and legend is not None:
        legend.remove()

    if save_config is not None and fig is not None:
        save_figure(fig, save_config)

    return fig, ax
