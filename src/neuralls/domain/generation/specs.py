"""Immutable input specifications for dataset generation.

These DTOs replace the ~20 individually-named keyword arguments that were
re-declared at every layer of the generation call chain. They mirror the two
sections a generation config already has — ``[source]`` (where samples come
from) and ``[generation]`` (how they are produced) — so one object is threaded
down instead of each layer re-flattening the same field set.

The domain layer cannot import ``composition.generation.DataGenerationContext``
(that would invert the dependency rule), so ``DataGenerationContext`` projects
itself into a :class:`SourceSpec` via its ``source_spec()`` method rather than
either side duplicating the fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .data_types import NormalizeType
from .interfaces import TracingSolverCallable
from .source_streams import EnumerateBy


@dataclass(frozen=True)
class SourceSpec:
    """Where a generation run reads its samples from.

    Attributes:
        matrix_path: Path expression (file or glob) for the matrix stream.
        rhs_path: Optional path expression for the RHS vector stream.
        solution_path: Optional path expression for a pre-computed solution
            stream; when set, each binding loads its solution vector and passes
            it into mixture generation as a single-row archive seed.
        parameters_paths: Path expressions for parameter vector streams.
        sample_id_regex: Optional regex used to extract sample ids from filenames.
        enumerate_by: Optional sequential id-assignment strategy for glob sources.
        include_indices: Restrict every glob-based stream to exactly these sample ids.
        exclude_indices: Drop these sample ids from every glob-based stream.
    """

    matrix_path: str
    rhs_path: str | None = None
    solution_path: str | None = None
    parameters_paths: tuple[str, ...] = ()
    sample_id_regex: str | None = None
    enumerate_by: EnumerateBy | None = None
    include_indices: tuple[int, ...] | None = None
    exclude_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class MixtureSpec:
    """Strategy mixing and RNG controls for one mixture generation call.

    Attributes:
        counts: Explicit per-strategy sample counts.
        mix: Strategy proportions, used together with ``total``.
        total: Total sample count, required when ``mix`` is given.
        seed: Random seed for reproducibility.
        shuffle: Whether to shuffle the generated samples.
        strategy_overrides: Per-strategy configuration overrides (for example
            ``krylov_iters``, ``cg_iters`` or ``solutions_glob``).
        solver_overrides: Optional per-strategy tracing solver overrides.
    """

    counts: Mapping[str, int] | None = None
    mix: Mapping[str, float] | None = None
    total: int | None = None
    seed: int = 42
    shuffle: bool = True
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None
    solver_overrides: Mapping[str, TracingSolverCallable] | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """How a full dataset is assembled from a source.

    Attributes:
        mixture: Strategy mixing and RNG controls applied per binding.
        replacement: Whether multi-matrix allocation may reuse matrix bindings
            for strategies that support it.
        normalize: Normalization strategy applied to each matrix sample.
        matrix_norm_type: Norm used to report the dataset-level matrix norm.
    """

    mixture: MixtureSpec = field(default_factory=MixtureSpec)
    replacement: bool = False
    normalize: NormalizeType = "matrix"
    matrix_norm_type: str = "spectral"


__all__ = ["DatasetSpec", "MixtureSpec", "SourceSpec"]
