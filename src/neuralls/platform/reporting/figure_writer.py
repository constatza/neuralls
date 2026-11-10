"""The one place a report figure is titled, saved and closed."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuralls.platform.reporting.figure_formats import FigureTarget

_HEADLESS_BACKEND = "Agg"


def use_headless_backend() -> None:
    """Make matplotlib render off-screen, whatever display the machine has.

    Report figures are only written to files. Called where figures are created rather than when
    a plotting module is imported, so importing a module never switches the process-wide backend.
    Switching backends closes every open figure, so an already headless process is left alone.
    """
    if matplotlib.get_backend().lower() != _HEADLESS_BACKEND.lower():
        matplotlib.use(_HEADLESS_BACKEND, force=True)


def finish_group_figure(
    fig: Figure,
    ax: Axes,
    *,
    title: str,
    subtitle: str | None,
    target: FigureTarget | None,
    show: bool,
    saved_message: str,
) -> None:
    """Title, save and close/show a figure; the only place a report figure is written."""
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if subtitle:
        ax.set_title(subtitle, fontsize=9)
    fig.tight_layout()

    if target is not None:
        target.directory.mkdir(parents=True, exist_ok=True)
        for path in target.paths:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            logger.info(f"{saved_message}: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
