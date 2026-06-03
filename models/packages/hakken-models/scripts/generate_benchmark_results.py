# benchmark_visualization.py
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from hakken_models.results.entities import (
    LegendConfig,
    MetricInfo,
    ModelInfo,
    OriginalName,
    SaveFigureConfig,
)
from hakken_models.results.latex import format_df_for_latex
from hakken_models.results.plots import mpl_style, plot_benchmark_bars, plot_radar_chart

# Distinct colors per model (matplotlib hex); same order as position 1–9
_MODEL_COLORS = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive (tab10 #9)
)


def prepare_model_mapping() -> dict[OriginalName, ModelInfo]:
    """Hard-coded model display names, ordering, and colors."""
    return {
        "Random": ModelInfo("Random", 1, _MODEL_COLORS[0]),
        "KGE-ComplEX": ModelInfo("ComplEx", 2, _MODEL_COLORS[1]),
        "KNN": ModelInfo("KNN", 3, _MODEL_COLORS[2]),
        "MLP": ModelInfo("MLP", 4, _MODEL_COLORS[3]),
        "Rule_Based": ModelInfo("RuleBased", 5, _MODEL_COLORS[4]),
        "Agatha": ModelInfo("Agatha", 6, _MODEL_COLORS[5]),
        "TNodeEmbed": ModelInfo("TNodeEmbed", 7, _MODEL_COLORS[6]),
        "ThiGER": ModelInfo("THiGER", 8, _MODEL_COLORS[7]),
        "ThiGERLLM": ModelInfo("THiGERLLM", 9, _MODEL_COLORS[8]),
    }


CONFIG_PATH = "../configs"

# Matplotlib qualitative colormaps (good for distinct model colors)
PALETTE_OPTIONS = (
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "Set1",
    "Set2",
    "Set3",
    "Paired",
    "Pastel1",
    "Pastel2",
    "Dark2",
    "Accent",
)


def _colors_for_models(
    model_names: list[str], model_mapping: dict[OriginalName, ModelInfo], palette_name: str
):
    """One color per model name (same order as model_names). Uses ModelInfo.color."""
    if palette_name not in PALETTE_OPTIONS:
        palette_name = "tab20"
    cmap = plt.get_cmap(palette_name)
    colors = []
    for model_name in model_names:
        original = next(
            (k for k, v in model_mapping.items() if v.display_name == model_name),
            None,
        )
        if original and original in model_mapping:
            colors.append(model_mapping[original].color)
        else:
            colors.append(cmap(0.05))
    return colors


@hydra.main(config_path=CONFIG_PATH, config_name="benchmark_viz", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point — reads CSV → formats table → saves LaTeX & radar chart
    """
    root = Path(cfg.paths.root)
    input_csv = root / cfg.paths.results_csv
    output_dir = root / cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading benchmark results from: {input_csv}")

    df_raw = pd.read_csv(input_csv, sep=",")

    model_mapping = prepare_model_mapping()
    metric_mapping_table = {}
    for m in cfg.benchmark_table_metrics:
        key = m.key
        display = m.get("display", key)
        higher = m.get("higher_is_better", True)
        metric_mapping_table[key] = MetricInfo(display, higher)

    metric_mapping_plot = {}
    for m in cfg.benchmark_plot_metrics:
        key = m.key
        display = m.get("display", key)
        higher = m.get("higher_is_better", True)  # usually ignored for radar
        metric_mapping_plot[key] = MetricInfo(display, higher)

    # ── Columns we want to keep ───────────────────────────────────────

    wanted_cols_table = ["model"] + list(metric_mapping_table.keys())
    df_table = df_raw[[c for c in wanted_cols_table if c in df_raw.columns]].copy()

    display_names_table = {"model": r"\textbf{Model}"}
    display_names_table.update({k: v.display_name for k, v in metric_mapping_table.items()})

    # ── LaTeX table preparation ───────────────────────────────────────
    latex_df = format_df_for_latex(
        df_table,
        metric_mapping=metric_mapping_table,
        decimals=cfg.latex.decimals,
        multiplier=cfg.latex.multiplier,
        rename_cols=display_names_table,
        model_name_mapping=model_mapping,
        format_metrics=True,
    )

    table_path = output_dir / f"benchmark_table_{cfg.paths.label}.txt"

    logger.info(f"Saving LaTeX table to: {table_path}")

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=False, column_format=cfg.latex.column_format))

    # ── Radar chart preparation ───────────────────────────────────────
    wanted_cols_plot = ["model"] + list(metric_mapping_plot.keys())
    df_plot = df_raw[[c for c in wanted_cols_plot if c in df_raw.columns]].copy()

    display_names_plot = {"model": "Model"}  # usually plain text for plots
    display_names_plot.update({k: v.display_name for k, v in metric_mapping_plot.items()})

    plot_df = format_df_for_latex(
        df_plot,
        metric_mapping=metric_mapping_plot,
        decimals=2,
        rename_cols=display_names_plot,
        model_name_mapping=model_mapping,
        format_metrics=False,
    )
    plot_df.fillna(0.0, inplace=True)

    # Usually better to sort models by some score (e.g. macro F1)
    if cfg.radar.sort_by_metric:
        sort_col = cfg.radar.sort_by_metric
        if sort_col in plot_df.columns:
            plot_df = plot_df.sort_values(sort_col, ascending=False)

    model_col = display_names_plot["model"]
    palette_name = cfg.get("palette", "tab20")
    colors = _colors_for_models(plot_df[model_col].tolist(), model_mapping, palette_name)

    legend_cfg = LegendConfig(
        loc=cfg.legend.loc,
        bbox_to_anchor=cfg.legend.bbox_to_anchor,
        frameon=cfg.legend.frameon,
        ncol=cfg.legend.ncol,
    )

    radar_path = output_dir / f"benchmark_radar_{cfg.paths.label}.pdf"
    legend_path = output_dir / f"benchmark_radar_legend_{cfg.paths.label}.pdf"

    logger.info(f"Generating radar chart {radar_path}")

    with mpl_style("nature"):
        fig, ax = plot_radar_chart(
            plot_df,
            model_col=display_names_plot.get("model", "model"),
            normalize=cfg.radar.normalize,
            remove_legend=True,
            title=cfg.radar.title,
            legend_config=legend_cfg,
            save_config=SaveFigureConfig(
                path=str(radar_path),
                dpi=cfg.radar.dpi,
                bbox_inches="tight",
            ),
            save_legend_config=SaveFigureConfig(
                path=str(legend_path),
                dpi=cfg.radar.dpi,
                bbox_inches="tight",
            ),
            colors=colors,
        )
        plt.close(fig)  # we save via save_config → no need to show()

    # ── Bar plot (optional subset of benchmark_plot_metrics via bar.metrics) ──
    if cfg.get("bar", {}).get("enabled", False):
        bar_metrics_cfg = cfg.bar.get("metrics")
        bar_metric_keys = (
            list(bar_metrics_cfg)
            if bar_metrics_cfg is not None
            else list(metric_mapping_plot.keys())
        )
        wanted_cols_bar = ["model"] + [k for k in bar_metric_keys if k in df_raw.columns]
        if len(wanted_cols_bar) > 1:
            df_bar_raw = df_raw[wanted_cols_bar].copy()
            if cfg.bar.get("sort_by_metric") and cfg.bar.sort_by_metric in df_bar_raw.columns:
                df_bar_raw = df_bar_raw.sort_values(cfg.bar.sort_by_metric, ascending=False)
            metric_mapping_bar = {
                k: metric_mapping_plot[k] for k in bar_metric_keys if k in metric_mapping_plot
            }
            display_names_bar = {"model": "Model"}
            display_names_bar.update({k: v.display_name for k, v in metric_mapping_bar.items()})
            df_bar = format_df_for_latex(
                df_bar_raw,
                metric_mapping=metric_mapping_bar,
                decimals=2,
                rename_cols=display_names_bar,
                model_name_mapping=model_mapping,
                format_metrics=False,
                sort_rows=False,
            )
            df_bar.fillna(0.0, inplace=True)
            model_col_bar = "Model"
            bar_metric_cols = [c for c in df_bar.columns if c != model_col_bar]
            metric_labels_bar = {c: c for c in bar_metric_cols}
            palette_name = cfg.get("palette", "tab20")
            colors_bar = _colors_for_models(
                df_bar[model_col_bar].tolist(), model_mapping, palette_name
            )
            bar_path = output_dir / f"benchmark_bars_{cfg.paths.label}.pdf"
            logger.info(f"Generating bar chart {bar_path}")
            with mpl_style("nature"):
                fig_bar, _ = plot_benchmark_bars(
                    df_bar,
                    model_col=model_col_bar,
                    metric_cols=bar_metric_cols,
                    metric_labels=metric_labels_bar,
                    title=cfg.bar.get("title", ""),
                    multiplier=cfg.bar.get("multiplier", 100.0),
                    sort_by_metric=None,
                    colors=colors_bar,
                    figsize=tuple(cfg.bar.get("figsize", [12, 4])),
                    ncols=cfg.bar.get("ncols", 3),
                    shared_ylim=(
                        tuple(cfg.bar.ylim)
                        if cfg.bar.get("shared_ylim") and cfg.bar.get("ylim")
                        else cfg.bar.get("shared_ylim", False)
                    ),
                    legend_config=legend_cfg,
                    save_config=SaveFigureConfig(
                        path=str(bar_path),
                        dpi=cfg.bar.get("dpi", 140),
                        bbox_inches="tight",
                    ),
                    remove_legend=True,
                    show_x_labels=cfg.bar.get("show_x_labels", False),
                    show_y_label=cfg.bar.get("show_y_label", True),
                    show_title=cfg.bar.get("show_title", True),
                    title_fontsize=cfg.bar.get("title_fontsize", 24),
                    ytick_fontsize=cfg.bar.get("ytick_fontsize", None),
                )
                plt.close(fig_bar)
        else:
            logger.warning("Bar plot enabled but no metric columns selected; skipping.")

    logger.info(f"Done. Files saved in: {output_dir}")


if __name__ == "__main__":
    main()
