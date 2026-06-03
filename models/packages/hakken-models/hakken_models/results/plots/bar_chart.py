"""Bar chart for benchmark model comparison: one subplot per metric, models on x-axis."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hakken_models.results.entities import LegendConfig, SaveFigureConfig
from hakken_models.results.plots.legend import get_legend, save_legend
from hakken_models.results.plots.save_figure import save_figure


def plot_benchmark_bars(
    df: pd.DataFrame,
    model_col: str = "model",
    metric_cols: list[str] | None = None,
    metric_labels: dict[str, str] | None = None,
    title: str = "",
    multiplier: float = 1.0,
    sort_by_metric: str | None = None,
    colors: list | None = None,
    figsize: tuple[float, float] = (12, 4),
    ncols: int = 3,
    shared_ylim: bool | tuple[float, float] = False,
    legend_config: LegendConfig | None = None,
    save_config: SaveFigureConfig | None = None,
    save_legend_config: SaveFigureConfig | None = None,
    remove_legend: bool = False,
    dpi: int = 140,
    show_x_labels: bool = False,
    show_y_label: bool = True,
    show_title: bool = True,
    title_fontsize: float = 24,
    ytick_fontsize: float | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one subplot per metric with bars per model (x = models, y = value).

    Uses bar.metrics subset when provided; otherwise all metric columns in df
    (excluding model_col) are used. Model order can be set via sort_by_metric.

    If shared_ylim is True, all subplots use the same y-axis range (computed
    from the data). If shared_ylim is (ymin, ymax), that range is used for all.

    If show_x_labels is True, model names are shown on the x-axis; otherwise they
    are hidden (ticks remain; legend still shows model names when present).
    If show_y_label is False, the y-axis label is hidden on each subplot (title
    still shows the metric name when show_title is True). If show_title is False,
    subplot titles (metric names) are hidden. A light horizontal grid is drawn behind the bars.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    all_metric_cols = [c for c in df.columns if c != model_col]
    if not all_metric_cols:
        raise ValueError("No metric columns found.")

    if metric_cols is None:
        metric_cols = all_metric_cols
    else:
        metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        raise ValueError("No requested metric columns present in DataFrame.")

    plot_df = df.copy()
    plot_df[metric_cols] = plot_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    plot_df.fillna(0.0, inplace=True)

    if sort_by_metric and sort_by_metric in plot_df.columns:
        plot_df = plot_df.sort_values(sort_by_metric, ascending=False)

    models = plot_df[model_col].astype(str).tolist()
    n_metrics = len(metric_cols)
    nrows = math.ceil(n_metrics / ncols) if ncols else 1
    ncols_actual = min(ncols, n_metrics)

    default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if colors is None:
        colors = [default_color_cycle[i % len(default_color_cycle)] for i in range(len(models))]

    fig, axes_flat = plt.subplots(
        nrows, ncols_actual, figsize=figsize, squeeze=False, layout="tight"
    )
    axes = axes_flat.flatten()

    x = np.arange(len(models))
    bar_width = 0.7  # one bar per model per subplot
    for ax in axes:
        ax.set_visible(False)

    for idx, metric_col in enumerate(metric_cols):
        ax = axes[idx]
        ax.set_visible(True)
        values = (plot_df[metric_col].values * multiplier).tolist()
        for i, (model, val) in enumerate(zip(models, values, strict=False)):
            ax.bar(
                x[i],
                val,
                width=bar_width,
                color=colors[i]
                if i < len(colors)
                else default_color_cycle[i % len(default_color_cycle)],
                label=model,
                align="center",
            )
        ax.set_xticks(x)
        x_labels = models if show_x_labels else [""] * len(models)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor")
        label = (metric_labels or {}).get(metric_col, metric_col)
        if show_y_label:
            ax.set_ylabel(label)
        if show_title:
            ax.set_title(label, fontsize=title_fontsize)
        if ytick_fontsize is not None:
            ax.tick_params(axis="y", labelsize=ytick_fontsize)
        ax.grid(True, axis="y", alpha=0.35, linestyle="-", color="gray")
        ax.set_axisbelow(True)

    for idx in range(len(metric_cols), len(axes)):
        axes[idx].set_visible(False)

    # Optional: same y limits for all subplots
    if shared_ylim is not False:
        if isinstance(shared_ylim, tuple) and len(shared_ylim) == 2:
            ymin, ymax = shared_ylim
        else:
            global_vals = plot_df[metric_cols].values * multiplier
            ymin = float(np.nanmin(global_vals))
            ymax = float(np.nanmax(global_vals))
            margin = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
            ymin = max(0, ymin - margin)
            ymax = ymax + margin
        for idx in range(len(metric_cols)):
            axes[idx].set_ylim(ymin, ymax)
        # # When sharing y limits, show y-axis tick labels only on the leftmost subplot
        # for idx in range(1, len(metric_cols)):
        #     axes[idx].tick_params(axis="y", labelleft=False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    # Legend on first subplot only (all models same across subplots)
    ax_legend = axes[0]
    legend = get_legend(ax_legend, legend_config)

    if save_legend_config is not None and legend is not None:
        save_legend(ax_legend, save_legend_config, legend_config)

    if remove_legend and legend is not None:
        legend.remove()

    if save_config is not None:
        save_figure(fig, save_config, close=False)

    return fig, axes
