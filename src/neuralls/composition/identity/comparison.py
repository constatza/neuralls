"""Comparison-stage identity: everything that determines a comparison's result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from neuralls.domain.identity import StageIdentity
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.storage.dataset_digest import current_dataset_digest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.digest import Digest, canonical_digest, content_digest


def input_digest(path: Path) -> Digest:
    """Digest a comparison input: a generated dataset by content, any other file by bytes.

    Args:
        path (Path): Dataset directory (has a manifest) or a raw input file/directory.

    Returns:
        Digest: Path-independent digest of what the input contains.
    """
    if path.is_dir() and (path / DATASET_MANIFEST_FILENAME).exists():
        return current_dataset_digest(path)
    return content_digest(path)


def _digestible_params(params: Mapping[str, object] | None) -> dict[str, object] | None:
    """Replace file paths inside RHS-source params by digests of their content."""
    if params is None:
        return None
    return {
        key: input_digest(value) if isinstance(value, Path) else value
        for key, value in params.items()
    }


def comparison_identity(
    cfg: ComparisonConfig,
    specs: Sequence[PreconditionerConfig],
    checkpoint_digests: Mapping[str, Digest],
) -> StageIdentity:
    """Derive the identity of one comparison run.

    Args:
        cfg (ComparisonConfig): Resolved comparison config (solver params, data
            selection, RHS source).
        specs (Sequence[PreconditionerConfig]): The preconditioners that will
            actually run (after checkpoint resolution).
        checkpoint_digests (Mapping[str, Digest]): Content digest of every
            resolved checkpoint the specs depend on.

    Returns:
        StageIdentity: Changes iff solver/tolerance settings, data selection,
        RHS source, any preconditioner setting, the matrix/RHS data, or any
        checkpoint's content changes — static-only comparisons included.
    """
    data = cfg.general.data
    general = replace(
        cfg.general,
        data=replace(data, rhs_source_params=_digestible_params(data.rhs_source_params)),
    )
    return StageIdentity.build(
        "comparison",
        {
            "general": canonical_digest(general),
            "preconditioners": canonical_digest(tuple(specs)),
            "matrix": input_digest(Path(data.matrix_path)),
            "checkpoints": canonical_digest(dict(checkpoint_digests)),
        },
    )
