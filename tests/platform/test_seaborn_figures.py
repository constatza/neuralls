"""`plot_frame` hands seaborn a pandas frame with the polars frame's values, `inf` included."""

from __future__ import annotations

import math

import polars as pl
import pytest

from neuralls.platform.reporting.seaborn_figures import plot_frame


@pytest.fixture
def summary() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "method_id": ["cg", "pcg"],
            "n_solved": [3, 5],
            "mean_time_s": [math.inf, 0.25],
        }
    )


def test_plot_frame_keeps_columns_and_values(summary: pl.DataFrame) -> None:
    frame = plot_frame(summary)

    assert list(frame.columns) == summary.columns
    assert frame["method_id"].tolist() == ["cg", "pcg"]
    assert frame["n_solved"].tolist() == [3, 5]
    assert frame["mean_time_s"].tolist() == [math.inf, 0.25]
