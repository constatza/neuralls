"""The validated, typed view of a compare session's stored method artifacts.

This is the only boundary a report reads through: a malformed or inconsistent artifact set is
rejected here, so downstream code operates on validated types and never on raw files.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.platform.config.models.preconditioner import (
    PreconditionerConfig,
    parse_preconditioner_config,
)
from neuralls.platform.reporting.benchmark_events import (
    MethodBenchmarkArtifacts,
    read_convergence,
    read_planned_systems,
    read_presentation,
    read_setup_events,
    read_solve_attempts,
)
from neuralls.shared.types import DatasetId, PreconditionerName


@dataclass(frozen=True, slots=True)
class MethodDirectory:
    """A downloaded method artifact directory and the MLflow run it was logged in.

    ``config_digest`` is the digest the run was tagged with; the stored summary must agree, so
    artifacts copied from or overwritten by another run are rejected.
    """

    run_id: str
    path: Path
    config_digest: str


@dataclass(frozen=True, slots=True)
class StoredMethod:
    """One method's stored workload: its identity, planned systems and raw event tables.

    Attributes:
        run_id: The method child run the artifacts were logged in.
        name: The method's result identifier.
        config_digest: Digest of the method configuration, as recorded in its summary.
        config: The method configuration as persisted. Checkpoint and dataset inputs hold
            content digests, not the paths used at compute time; it serves plot-style
            derivation, never a rebuild.
        presentation: The method's scientific presentation with its realized facts.
        systems: Every planned system with its solver settings and reference state, in plan order.
        setup_events: Setup-event table.
        solve_attempts: Solve-attempt table, linked to ``setup_events``.
        convergence: Per-iteration convergence table.
    """

    run_id: str
    name: PreconditionerName
    config_digest: str
    config: PreconditionerConfig
    presentation: ExecutedMethod
    systems: tuple[SystemProvenance, ...]
    setup_events: pl.DataFrame
    solve_attempts: pl.DataFrame
    convergence: pl.DataFrame


@dataclass(frozen=True, slots=True)
class StoredBenchmark:
    """Every method of one compare session, with the presentation of each dataset they ran on."""

    methods: tuple[StoredMethod, ...]
    datasets: Mapping[DatasetId, DatasetPresentation]


def read_stored_method(
    source: MethodDirectory,
) -> tuple[StoredMethod, Mapping[DatasetId, DatasetPresentation]]:
    """Read and cross-validate one method's artifact directory.

    Returns the method and the presentations of the datasets it ran on.
    """
    directory = source.path
    artifacts = MethodBenchmarkArtifacts()
    setup_path = directory / artifacts.setup_events
    config = parse_preconditioner_config(
        tomllib.loads((directory / artifacts.method_config).read_text(encoding="utf-8"))
    )
    stored = read_presentation(directory / artifacts.presentation)
    systems = read_planned_systems(directory / artifacts.planned_systems)
    setup_events = read_setup_events(setup_path)
    solve_attempts = read_solve_attempts(
        directory / artifacts.solve_attempts, setup_events_path=setup_path
    )
    convergence = read_convergence(directory / artifacts.convergence)

    summary = tomllib.loads((directory / artifacts.summary).read_text(encoding="utf-8"))
    name = PreconditionerName(config.name)
    if str(summary["method"]["config_digest"]) != source.config_digest:
        raise ValueError(f"summary in {directory} does not match the digest of run {source.run_id}")
    if summary["method"]["name"] != str(name):
        raise ValueError(f"summary in {directory} does not belong to method {name}")
    if stored.method.prepared.method_id != name:
        raise ValueError(f"presentation in {directory} does not belong to method {name}")
    if {system.tag.dataset for system in systems} != set(stored.datasets):
        raise ValueError(f"dataset presentations in {directory} do not match its planned systems")
    planned_ids = {str(system.tag.system_id) for system in systems}
    for table, frame in (("solve attempts", solve_attempts), ("convergence", convergence)):
        if not set(frame.get_column("system_id").to_list()) <= planned_ids:
            raise ValueError(f"{table} in {directory} cover systems that were not planned")
    for table, frame in (
        ("setup events", setup_events),
        ("solve attempts", solve_attempts),
        ("convergence", convergence),
    ):
        if {str(value) for value in frame.get_column("method").to_list()} - {str(name)}:
            raise ValueError(f"{table} in {directory} belong to another method")
    method = StoredMethod(
        run_id=source.run_id,
        name=name,
        config_digest=str(summary["method"]["config_digest"]),
        config=config,
        presentation=stored.method,
        systems=systems,
        setup_events=setup_events,
        solve_attempts=solve_attempts,
        convergence=convergence,
    )
    return method, stored.datasets


def read_stored_benchmark(directories: Sequence[MethodDirectory]) -> StoredBenchmark:
    """Read the artifact directory of every method of a session into one validated benchmark."""
    if not directories:
        raise ValueError("a stored benchmark needs at least one method")
    methods: dict[PreconditionerName, StoredMethod] = {}
    datasets: dict[DatasetId, DatasetPresentation] = {}
    for directory in directories:
        method, method_datasets = read_stored_method(directory)
        if method.name in methods:
            raise ValueError(f"method {method.name} is stored more than once")
        methods[method.name] = method
        for dataset, presentation in method_datasets.items():
            if datasets.setdefault(dataset, presentation) != presentation:
                raise ValueError(f"methods present dataset {dataset} differently")
    return StoredBenchmark(methods=tuple(methods.values()), datasets=datasets)
