import matplotlib.pyplot as plt
from loguru import logger

from hakken_models.results.entities import SaveFigureConfig


def save_figure(
    fig: plt.Figure,
    save_config: SaveFigureConfig,
    close: bool = True,
) -> None:
    """
    Save the given figure according to the provided SaveFigureConfig.

    Parameters:
      - fig: The matplotlib Figure to save.
      - save_config: Configuration for saving the figure, including path and format.
    """

    fig.savefig(
        save_config.path,
        dpi=save_config.dpi,
        bbox_inches=save_config.bbox_inches,
    )
    logger.info(f"Saved figure to {save_config.path}")
    if close:
        plt.close(fig)
