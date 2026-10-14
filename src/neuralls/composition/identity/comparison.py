"""Comparison-stage identity: everything that determines a comparison's result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from neuralls.domain.identity import StageIdentity
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.storage.dataset_digest import current_dataset_digest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME, SPLIT_FILENAME
from neuralls.shared.digest import Digest, canonical_digest, content_digest, memoized


def input_digest(path: Path) -> Digest:
    """Digest a comparison input: a generated dataset by content, any other file by bytes.

    Args:
        path (Path): Dataset directory (has a manifest; its pinned split, when present,
            is part of the digest) or a raw input file/directory.

    Returns:
        Digest: Path-independent digest of what the input contains.
    """
    if path.is_dir() and (path / DATASET_MANIFEST_FILENAME).exists():
        return memoized(("dataset", path.resolve()), lambda: _dataset_input_digest(path))
    return content_digest(path)


def _dataset_input_digest(path: Path) -> Digest:
    split_path = path / SPLIT_FILENAME
    if not split_path.exists():
        return current_dataset_digest(path)
    return canonical_digest(current_dataset_digest(path), content_digest(split_path))


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


def preconditioner_identities(
    specs: Sequence[PreconditionerConfig],
    checkpoint_digests: Mapping[str, Digest],
) -> dict[str, Digest]:
    """Identity of each preconditioner, keyed by its (display) name.

    A preconditioner's name is a readable label derived from its config; for a
    neural one it names the model *reference*, not the model, so two different
    checkpoints can share it. This is the key that actually decides whether two
    same-named preconditioners are the same: the config's identity digest
    (excluding resolved local paths) plus the content digest of every
    checkpoint it loads. Two specs with equal identity are interchangeable even
    if their resolved temp paths differ; equal names with different identity are
    different preconditioners.

    Args:
        specs (Sequence[PreconditionerConfig]): Specs after checkpoint resolution.
        checkpoint_digests (Mapping[str, Digest]): Per-checkpoint content
            digests keyed ``"<spec index>:<label>"``, as the resolution step
            produces them.

    Returns:
        dict[str, Digest]: Preconditioner name -> identity digest.
    """
    return {
        spec.name: canonical_digest(
            {
                "spec": canonical_digest(spec),
                "checkpoints": canonical_digest(
                    {
                        key: digest
                        for key, digest in checkpoint_digests.items()
                        if key.startswith(f"{index}:")
                    }
                ),
            }
        )
        for index, spec in enumerate(specs)
    }
