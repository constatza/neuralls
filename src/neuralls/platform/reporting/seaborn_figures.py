"""Seaborn entry point for report figures: the one polars-to-pandas conversion."""

from __future__ import annotations

import pandas as pd
import polars as pl


def plot_frame(df: pl.DataFrame) -> pd.DataFrame:
    """The summary frame as seaborn reads it.

    Seaborn 0.13 raises `TypeError` on a polars frame, so every seaborn call takes its
    data from here; summary marks (bar heights, whiskers) are still read from the
    polars summary frame that is stored as the figure's table.
    """
    return df.to_pandas()
