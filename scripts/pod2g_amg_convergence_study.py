r"""POD-2G vs AMG-aggregation convergence/wall-time diagnostic (docs/plan.md §D/§Verification).

Sweeps AMG's ``theta`` and POD-2G's ``rank`` independently, records each run's
*realized* coarse dimension (never the input hyperparameter — both are
threshold-driven with an emergent dimension, docs/plan.md §C.1), and reports
three views, all built from measured wall-clock time on this machine rather
than an analytic FLOP model: a formula-based cost estimate never changes
across runs/hardware and belongs in a paper, not in something you'd trust to
tell you which preconditioner is actually faster here, today.

  (i)   iterations-to-converge vs. realized coarse dimension c
  (ii)  cold-start wall-clock time (setup + solve) vs. realized coarse dim
  (iii) amortized wall-clock time per solve vs. K, for the AMG/POD-2G point
        pair whose realized coarse dimensions are closest to each other
        (docs/plan.md §C.2's "match by measurement, not by construction"),
        so the POD-2G/AMG break-even K is visible

Timing is two composable decorators rather than an ad hoc wrapper: ``timed``
turns any callable into one returning ``(elapsed_seconds, result)``, and
``fastest_of`` repeats a ``timed`` callable ``--repeats`` times and keeps the
fastest run — the standard microbenchmark practice of discounting
OS/scheduler noise (noise can only slow a run down, never speed it up),
directly addressing docs/plan.md's own caution that "wall-clock on dense
small-n matrices is noisy." There is no per-iteration timing breakdown:
``run_cg_comparison`` does not instrument individual CG iterations, so
residual-vs-time curves within a single solve are not reported here (that
would require touching the solver's internals, out of scope for a
diagnostic script).

Example:
    uv run python scripts/pod2g_amg_convergence_study.py \\
        --matrix-dir "$NEURALLS_PROCESSED_DIR/gaussian-eval-45x15randomE" \\
        --snapshot-dir "$NEURALLS_PROCESSED_DIR/gaussian-cg50-45x15randomE" \\
        --output-dir outputs/pod2g_amg_study/45x15randomE
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
from torchalg.preconditioners.implementations.amg import TargetDimensionCoarsening

from neuralls.composition.preconditioners.factory import create_preconditioner_with_coarsening
from neuralls.domain.solver.comparison import run_cg_comparison
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    PODCoarseningConfig,
    TargetDimCoarseningConfig,
)
from neuralls.platform.storage.comparison import load_system_arrays
from neuralls.platform.storage.dataset_readers import load_dense_training_arrays

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy

    from neuralls.domain.solver.models.cg_result import CGComparisonResult


def _realized_coarse_dimension(coarsening: CoarseningStrategy, matrix: torch.Tensor) -> int:
    """Read back the actual coarse dimension a coarsening strategy produces.

    AMG's `theta` and POD-2G's `rank` (often a float energy threshold) are
    only ever emergent dimensions, never directly chosen — this reads the
    built result instead of trusting either hyperparameter.
    """
    a_coarse, _ = coarsening.build_transfer(matrix)
    return int(a_coarse.shape[0])


@dataclass(frozen=True)
class SweepPoint:
    """One (method, hyperparameter) sweep result at its realized coarse dimension.

    Attributes:
        method: Display label, e.g. ``"amg (theta=0.05)"`` or ``"POD-2G (rank=20)"``.
        coarse_dim: Realized coarse dimension c (never the input hyperparameter).
        result: CG solver outcome (iterations, residual history) for this run.
        setup_seconds: Measured wall-clock time to build/fit the preconditioner.
        solve_seconds: Measured wall-clock time for the CG solve itself.
    """

    method: str
    coarse_dim: int
    result: CGComparisonResult
    setup_seconds: float
    solve_seconds: float

    def amortized_seconds(self, amortize_over: int) -> float:
        """Wall-clock time per solve when ``setup_seconds`` is shared over K solves."""
        return self.setup_seconds / amortize_over + self.solve_seconds


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, tuple[float, R]]:
    """Wrap ``fn`` to return ``(elapsed_seconds, result)`` instead of just its result."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[float, R]:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        return time.perf_counter() - start, result

    return wrapper


def fastest_of[**P, R](
    fn: Callable[P, tuple[float, R]], repeats: int
) -> Callable[P, tuple[float, R]]:
    """Wrap a ``timed`` function to call it ``repeats`` times and keep the fastest run.

    Standard microbenchmark practice: OS/scheduler noise can only slow a run
    down, never speed it up, so the minimum elapsed time over independent
    repeats is the least-noisy estimate — this is what ``timeit`` itself
    recommends, though ``timeit`` discards each call's return value, which
    this script needs (the built preconditioner, the solve's residual
    history), so it's reimplemented as a decorator here instead.
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[float, R]:
        return min((fn(*args, **kwargs) for _ in range(repeats)), key=lambda run: run[0])

    return wrapper


def _sweep_amg(
    A: torch.Tensor,
    b: torch.Tensor,
    thetas: list[float],
    *,
    repeats: int,
    maxiter: int,
    n_pre: int,
    n_post: int,
    smoother_omega: float,
    n_levels: int,
) -> list[SweepPoint]:
    """Run one CG comparison per AMG-aggregation ``theta``, in ascending realized c."""
    points = []
    for theta in thetas:
        config = AMGPreconditionerConfig(
            n_levels=n_levels,
            pre_smoothing_steps=n_pre,
            post_smoothing_steps=n_post,
            smoother_omega=smoother_omega,
            coarsening=AggregationCoarseningConfig(theta=theta, omega=smoother_omega),
        )
        timed_build = timed(create_preconditioner_with_coarsening)
        build = fastest_of(timed_build, repeats)
        setup_seconds, (precond, coarsening) = build(A, config)
        assert coarsening is not None, "AMG-type config must build a coarsening strategy"
        coarse_dim = _realized_coarse_dimension(coarsening, A)
        timed_solve = timed(run_cg_comparison)
        solve = fastest_of(timed_solve, repeats)
        solve_seconds, results = solve(
            A, b, preconditioners={config.name: precond}, maxiter=maxiter
        )
        points.append(
            SweepPoint(config.name, coarse_dim, results[config.name], setup_seconds, solve_seconds)
        )
    return sorted(points, key=lambda p: p.coarse_dim)


def _sweep_pod(
    A: torch.Tensor,
    b: torch.Tensor,
    ranks: list[int],
    *,
    snapshot_dir: Path,
    n_snapshots: int,
    repeats: int,
    maxiter: int,
    n_pre: int,
    n_post: int,
    smoother_omega: float,
    n_levels: int,
) -> list[SweepPoint]:
    """Run one CG comparison per POD-2G ``rank``, fit inline from ``snapshot_dir``.

    ``setup_seconds`` measures the real SVD/fit call this script performs —
    it does not include the offline CG-based snapshot-generation cost that
    already happened once, earlier, when the dataset in ``snapshot_dir`` was
    generated (that cost was paid on whatever machine ran `generate`, not
    this one, and isn't reproducible from a script that only reads its
    output arrays).
    """
    points = []
    for rank in ranks:
        config = AMGPreconditionerConfig(
            n_levels=n_levels,
            pre_smoothing_steps=n_pre,
            post_smoothing_steps=n_post,
            smoother_omega=smoother_omega,
            coarsening=PODCoarseningConfig(
                dataset_dir=snapshot_dir, rank=rank, n_snapshots=n_snapshots
            ),
        )
        timed_build = timed(create_preconditioner_with_coarsening)
        build = fastest_of(timed_build, repeats)
        setup_seconds, (precond, coarsening) = build(A, config)
        assert coarsening is not None, "AMG-type config must build a coarsening strategy"
        coarse_dim = _realized_coarse_dimension(coarsening, A)
        timed_solve = timed(run_cg_comparison)
        solve = fastest_of(timed_solve, repeats)
        solve_seconds, results = solve(
            A, b, preconditioners={config.name: precond}, maxiter=maxiter
        )
        points.append(
            SweepPoint(config.name, coarse_dim, results[config.name], setup_seconds, solve_seconds)
        )
    return sorted(points, key=lambda p: p.coarse_dim)


def _closest_match(
    amg_points: list[SweepPoint], pod_points: list[SweepPoint]
) -> tuple[SweepPoint, SweepPoint]:
    """Pick the (amg, pod2g) pair with the closest realized coarse dimensions."""
    return min(
        ((a, p) for a in amg_points for p in pod_points),
        key=lambda pair: abs(pair[0].coarse_dim - pair[1].coarse_dim),
    )


def _save_plot(fig: plt.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")
    plt.close(fig)


def _plot_iterations_vs_coarse_dim(
    amg_points: list[SweepPoint], pod_points: list[SweepPoint], save_path: Path
) -> None:
    """Plot (i): iterations-to-converge vs. realized coarse dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [p.coarse_dim for p in amg_points],
        [p.result.iterations for p in amg_points],
        "o-",
        label="AMG (aggregation)",
    )
    ax.plot(
        [p.coarse_dim for p in pod_points],
        [p.result.iterations for p in pod_points],
        "s-",
        label="POD-2G",
    )
    ax.set_xlabel("Realized coarse dimension c")
    ax.set_ylabel("CG iterations to converge")
    ax.set_title("Coarse-space quality at equal budget (wall-time excluded)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    _save_plot(fig, save_path)


def _plot_time_vs_coarse_dim(
    amg_points: list[SweepPoint], pod_points: list[SweepPoint], save_path: Path
) -> None:
    """Plot (ii): measured cold-start wall-clock time vs. realized coarse dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [p.coarse_dim for p in amg_points],
        [p.setup_seconds + p.solve_seconds for p in amg_points],
        "o-",
        label="AMG (aggregation)",
    )
    ax.plot(
        [p.coarse_dim for p in pod_points],
        [p.setup_seconds + p.solve_seconds for p in pod_points],
        "s-",
        label="POD-2G",
    )
    ax.set_xlabel("Realized coarse dimension c")
    ax.set_ylabel("Wall-clock time (seconds): setup + solve, cold start")
    ax.set_title("Measured cold-start cost (min over repeats)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    _save_plot(fig, save_path)


def _plot_amortized_time_vs_k(
    amg_point: SweepPoint, pod_point: SweepPoint, k_values: list[int], save_path: Path
) -> None:
    """Plot (iii): amortized wall-clock time per solve vs. K, at matched realized c."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        k_values,
        [amg_point.amortized_seconds(k) for k in k_values],
        "o-",
        label=f"{amg_point.method}, c={amg_point.coarse_dim}",
    )
    ax.plot(
        k_values,
        [pod_point.amortized_seconds(k) for k in k_values],
        "s-",
        label=f"{pod_point.method}, c={pod_point.coarse_dim}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("K (solves sharing one built/fitted preconditioner)")
    ax.set_ylabel("Amortized wall-clock time per solve (seconds)")
    ax.set_title("Break-even point: setup cost amortized over K reuses")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    _save_plot(fig, save_path)


def _parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated list of floats, e.g. ``"0.05,0.1"`` -> ``[0.05, 0.1]``."""
    return [float(v) for v in text.split(",") if v.strip()]


def _parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated list of ints, e.g. ``"1,10,100"`` -> ``[1, 10, 100]``."""
    return [int(v) for v in text.split(",") if v.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Dataset dir supplying the comparison matrix + RHS.",
    )
    parser.add_argument("--matrix-index", type=int, default=0)
    parser.add_argument("--rhs-index", type=int, default=0)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="Dataset dir whose `solutions` array supplies POD-2G training snapshots.",
    )
    parser.add_argument("--n-snapshots", type=int, default=-1, help="-1 means use all snapshots.")
    parser.add_argument("--theta-grid", type=str, default="0.02,0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--rank-grid", type=str, default="2,5,10,20,40,80")
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,5,10,20,50,100",
        help="Amortization sweep for plot (iii).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent timing repeats per point; the minimum is kept to discount OS/scheduler noise.",
    )
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--n-pre", type=int, default=2)
    parser.add_argument("--n-post", type=int, default=2)
    parser.add_argument("--smoother-omega", type=float, default=0.67)
    parser.add_argument("--n-levels", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pod2g_amg_study"))
    parser.add_argument(
        "--match-pod-rank",
        type=int,
        default=None,
        help=(
            "If set, fit POD-2G at this rank, read its realized coarse dimension, "
            "then use TargetDimensionCoarsening (torchalg) to brute-force search a "
            "theta grid for the AMG-aggregation theta whose realized c is closest, "
            "and print the matched pair."
        ),
    )
    parser.add_argument(
        "--theta-step",
        type=float,
        default=0.01,
        help="Grid spacing for --match-pod-rank's brute-force theta search.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    a_np, b_np = load_system_arrays(
        args.matrix_dir,
        args.matrix_dir,
        rhs_sample_index=args.rhs_index,
        matrix_index=args.matrix_index,
    )
    A = torch.tensor(a_np, dtype=torch.float64)
    b = torch.tensor(b_np, dtype=torch.float64)

    _, solutions = load_dense_training_arrays(args.snapshot_dir)
    if args.n_snapshots != -1:
        solutions = solutions[: args.n_snapshots]

    sweep_kwargs = {
        "repeats": args.repeats,
        "maxiter": args.maxiter,
        "n_pre": args.n_pre,
        "n_post": args.n_post,
        "smoother_omega": args.smoother_omega,
        "n_levels": args.n_levels,
    }
    amg_points = _sweep_amg(A, b, _parse_float_list(args.theta_grid), **sweep_kwargs)
    pod_points = _sweep_pod(
        A,
        b,
        _parse_int_list(args.rank_grid),
        snapshot_dir=args.snapshot_dir,
        n_snapshots=args.n_snapshots,
        **sweep_kwargs,
    )

    for p in (*amg_points, *pod_points):
        print(
            f"{p.method}: c={p.coarse_dim}, iterations={p.result.iterations}, "
            f"setup={p.setup_seconds:.4g}s, solve={p.solve_seconds:.4g}s"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_iterations_vs_coarse_dim(
        amg_points, pod_points, args.output_dir / "iterations_vs_coarse_dim.png"
    )
    _plot_time_vs_coarse_dim(amg_points, pod_points, args.output_dir / "walltime_vs_coarse_dim.png")

    amg_match, pod_match = _closest_match(amg_points, pod_points)
    _plot_amortized_time_vs_k(
        amg_match,
        pod_match,
        _parse_int_list(args.k_values),
        args.output_dir / "walltime_amortized_vs_k.png",
    )

    if args.match_pod_rank is not None:
        pod_config = AMGPreconditionerConfig(
            n_levels=args.n_levels,
            pre_smoothing_steps=args.n_pre,
            post_smoothing_steps=args.n_post,
            smoother_omega=args.smoother_omega,
            coarsening=PODCoarseningConfig(
                dataset_dir=args.snapshot_dir,
                rank=args.match_pod_rank,
                n_snapshots=args.n_snapshots,
            ),
        )
        _, pod_coarsening = create_preconditioner_with_coarsening(A, pod_config)
        assert pod_coarsening is not None
        target_c = _realized_coarse_dimension(pod_coarsening, A)

        target_dim_config = AMGPreconditionerConfig(
            n_levels=args.n_levels,
            pre_smoothing_steps=args.n_pre,
            post_smoothing_steps=args.n_post,
            smoother_omega=args.smoother_omega,
            coarsening=TargetDimCoarseningConfig(
                target_coarse_dim=target_c, step=args.theta_step, omega=args.smoother_omega
            ),
        )
        _, matched_coarsening = create_preconditioner_with_coarsening(A, target_dim_config)
        assert matched_coarsening is not None
        matched_c = _realized_coarse_dimension(matched_coarsening, A)
        assert isinstance(matched_coarsening, TargetDimensionCoarsening)
        theta = matched_coarsening._theta
        print(
            f"Brute-force match: POD-2G rank={args.match_pod_rank} (c={target_c}) "
            f"<-> amg theta={theta:g} (c={matched_c})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
