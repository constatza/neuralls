"""Shared type aliases for tests that build prepared method presentations."""

from __future__ import annotations

from collections.abc import Callable

from neuralls.shared.scientific_description import PreparedMethodPresentation

type PreparedMethodFactory = Callable[[str], PreparedMethodPresentation]
"""The `prepared_method` fixture: the validated presentation of one method id."""
