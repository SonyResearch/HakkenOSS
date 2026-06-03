import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend

from hakken_models.results.entities import LegendConfig, SaveFigureConfig
from hakken_models.results.plots.save_figure import save_figure


def get_legend_kwargs(config: LegendConfig | None) -> dict:
    if config is None:
        return {}

    legend_kwargs = {
        "loc": config.loc,
        "bbox_to_anchor": config.bbox_to_anchor,
        "frameon": config.frameon,
    }

    if config.ncol is not None:
        legend_kwargs["ncol"] = config.ncol

    return legend_kwargs


def get_legend(
    ax: Axes,
    config: LegendConfig | None = None,
) -> Legend:
    """TODO"""

    if config is None:
        config = LegendConfig()

    legend_kwargs = get_legend_kwargs(config)
    return ax.legend(**legend_kwargs)


def save_legend(
    ax: Axes,
    save_config: SaveFigureConfig,
    config: LegendConfig | None = None,
) -> None:
    """
    Save the legend of a given axis as a standalone figure.
    """
    legend_kwargs = get_legend_kwargs(config)

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        raise ValueError("No legend handles found on axis.")

    fig_leg = plt.figure(figsize=(4, 4))  # temporary size, will be resized tightly
    fig_leg.canvas.draw()

    legend: Legend = fig_leg.legend(
        handles,
        labels,
        **legend_kwargs,
    )

    # Force draw so we can measure legend bbox
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent(fig_leg.canvas.get_renderer())  # type: ignore

    # Convert bbox from pixels to inches
    bbox_inches = bbox.transformed(fig_leg.dpi_scale_trans.inverted())

    # Resize figure to legend size
    fig_leg.set_size_inches(bbox_inches.width, bbox_inches.height)

    # Save tightly
    save_figure(fig_leg, save_config)

    plt.close(fig_leg)
