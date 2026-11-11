"""Pydantic boundary config for how report figures are laid out.

Complete defaults mean an ordinary config needs no ``[figure_style]`` section at all.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuralls.shared.constants import (
    DEFAULT_LABEL_ROTATION_DEG,
    DEFAULT_MIN_BOX_SAMPLES,
    DEFAULT_MIN_VIOLIN_SAMPLES,
    MAX_LABEL_ROTATION_DEG,
)

__all__ = ["FigureStyleConfig"]


class FigureStyleConfig(BaseModel):
    """``[figure_style]`` TOML section controlling report figure layout.

    Example::

        [figure_style]
        label_rotation_deg = 45.0
        min_box_samples = 5
        min_violin_samples = 30
        show_titles = false

    A distribution figure draws a method's systems as points only below ``min_box_samples``, as
    points over a box from there, and as a violin over a box from ``min_violin_samples`` on.
    ``show_titles = false`` draws every group and per-system figure without its title and subtitle;
    the figure card keeps the title.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_rotation_deg: float = Field(
        default=DEFAULT_LABEL_ROTATION_DEG, ge=0.0, le=MAX_LABEL_ROTATION_DEG
    )

    min_box_samples: int = Field(default=DEFAULT_MIN_BOX_SAMPLES, ge=1)
    min_violin_samples: int = Field(default=DEFAULT_MIN_VIOLIN_SAMPLES, ge=1)

    show_titles: bool = True

    @model_validator(mode="after")
    def _box_before_violin(self) -> Self:
        if self.min_box_samples > self.min_violin_samples:
            raise ValueError(
                f"min_box_samples ({self.min_box_samples}) must not exceed "
                f"min_violin_samples ({self.min_violin_samples})"
            )
        return self
