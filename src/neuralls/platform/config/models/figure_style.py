"""Pydantic boundary config for how report figures are laid out.

Complete defaults mean an ordinary config needs no ``[figure_style]`` section at all.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from neuralls.shared.constants import DEFAULT_LABEL_ROTATION_DEG, MAX_LABEL_ROTATION_DEG

__all__ = ["FigureStyleConfig"]


class FigureStyleConfig(BaseModel):
    """``[figure_style]`` TOML section controlling report figure layout.

    Example::

        [figure_style]
        label_rotation_deg = 45.0
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_rotation_deg: float = Field(
        default=DEFAULT_LABEL_ROTATION_DEG, ge=0.0, le=MAX_LABEL_ROTATION_DEG
    )
