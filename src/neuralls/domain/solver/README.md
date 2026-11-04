# Solver Workflow Boundary

`neuralls.domain.solver` owns comparison DTOs and pure reporting reductions.
CG, PCG, FCG, monitoring primitives, solver results, and preconditioner
implementations are delegated to `torchalg`.

## Canonical benchmark data

`models/benchmark_events.py` defines the canonical online measurements:

- `MatrixSetupEvent` records one method setup for one matrix.
- `SolveAttempt` records one RHS attempt and requires its `SetupEventId`.
- setup and solve success/failure are closed variants;
- peak memory is either measured or explicitly unknown;
- representative seconds and every repeat sample are retained separately.

`run_cg_comparison` uses the failure-aware repeat timer, so a solver exception
still yields a measured failed attempt with its representative duration,
samples, and peak memory rather than an unmeasured zero-cost row.

Online time is setup plus solve only. Checkpoint and dataset reads, reference
solves, staging, plotting, persistence, MLflow calls, and historical generation
or training remain outside it. Required computation after I/O belongs to a
timed setup or solve event.

`comparison_summary.py::format_results_summary` heads each summary line with the method's description, rendered from the `ConfiguredMethodPresentation` it is given for every result (a result without one raises `MissingMethodPresentationError`).

`models/workload.py` represents a method summary as either complete or
explicitly incomplete. Both variants retain planned, attempted, and converged
RHS counts, setup and solve totals, total online workload time, and workload
peak memory. Incomplete summaries name the
unattempted systems and cannot be ranked.

## Event reductions

`reporting/event_frames.py` is the Polars serialization boundary. Nullable
columns encode closed event variants only in normalized frames; core domain
objects remain variant-based.

`validate_setup_solve_frame_links` is the shared frame-level reuse check: each
solve links to exactly one completed setup with matching dataset, matrix, and
method identity. Platform Parquet readers call the same validator before an
artifact can be reused.

The setup and solve Parquet frames are canonical for realized event timing,
repeat samples, outcomes, per-event memory, and attempted/converged counts.
They intentionally do not persist the planned `SystemTag` sequence or the
method-lifetime peak observed by the Task 2 reducer. `summary.toml` is therefore
canonical for planned membership, missing system IDs, and workload-lifetime
memory; those values cannot be reconstructed from only the two event frames.

`reporting/aggregation.py` validates planned identities and solve-to-setup
links before deriving costs. Planned RHS membership is counted by
`(dataset, matrix_id)` before attempts are joined:

```text
allocated_setup_s = setup_s / planned_rhs_count
amortized_online_s = solve_s + allocated_setup_s
matrix_online_s = setup_s + sum(solve_s)
workload_online_s = sum(matrix_online_s)
```

One matrix setup is therefore charged once even when sibling RHS values belong
to different recipes. Matrix-set-and-RHS-type summaries allocate that setup using the
complete matrix membership; matrix-set and all-matrix-sets summaries retain the whole setup
once. Workload totals use `math.fsum` over representative event values;
warmups and individual repeat samples are never added to totals.

Two quantities divide a matrix's setup over its planned RHS and are named apart. `online_s` of the
observation view uses the matrix's whole planned RHS count $r_m$, whichever report group the system
is in. `allocated_setup_s`, the setup field of `CompleteWorkloadSummary` and
`IncompleteWorkloadSummary` and the matching column of the workload-totals table, allocates setup
to one report group by that group's share of the matrix's RHS, so the groups of a matrix add up to
one setup. Neither is a measurement; both are shares of one measured setup.

`reporting/classification.py` consumes planned tags, setup events, solve
attempts, and a report level. A `ConvergenceSummary` keeps planned, attempted,
and converged counts distinct and reports iteration statistics over converged
attempts only. Explicit per-method planned membership retains an interrupted
method in every recipe/dataset group even when execution never reached that
group's first setup. Every `GroupAggregated` report carries `observations`, the group's rows of the
all-methods observation view (`classify_level` receives the view of the whole workload and keeps the
rows of the group's systems): a `RecipeProfileReport` holds its one recipe's rows, a
`WorkloadMixtureReport` the pooled rows of every recipe it names. The recipe-versus-mixture split
only gates which figure kinds apply, because a performance profile and a paired comparison need
one homogeneous recipe. Failed attempts keep their measured time in the observation frame and in
workload totals, and every per-system metric excludes them through `select`.

## Observation view and metric catalog

`reporting/observations.py::observations(method, planned_systems, setup_events, solve_attempts, convergence)`
joins the four stored frames of one method into one row per planned system
(`OBSERVATION_SCHEMA`); `observations_of_methods` concatenates methods, sorted by method,
dataset, matrix and RHS ordinal. The view is computed in memory from the stored parquets and is
never persisted, so it needs no `compare` re-run. The inputs are validated first: a frame of
another method, a duplicated or unplanned system, a solve that contradicts its planned system and
a solve linked to a failed or foreign setup, and a convergence history of another method or of an
unplanned system raise `ValueError`.

| Column | Source |
|---|---|
| `dataset`, `recipe`, `matrix_id`, `system_id`, `rhs_ordinal` | `planned_systems` |
| `method` | the `method` argument (`planned_systems` has no method column) |
| `outcome` | `Outcome` value: `solve_outcome_expr` on the `solve_attempts` row; no row and `setup_events.setup_status = failed` is `setup_failed`; otherwise `not_attempted` |
| `iterations` | `solve_attempts.iterations`; null unless the outcome is `converged` or `unconverged` |
| `solve_s` | `solve_attempts.solve_s`; null without a solve row |
| `setup_s` | `setup_events.setup_s` of the system's `(dataset, matrix_id)`; null without a setup row |
| `planned_rhs` | planned systems of the system's `(dataset, matrix_id)` over every recipe in `planned_systems`, the count the setup is divided by |
| `online_s`, `standalone_s` | `solve_s + setup_s / planned_rhs` and `solve_s + setup_s`; null when either term is null |
| `time_per_iteration_s` | `solve_s / iterations` for `converged` and `unconverged` rows with at least one iteration |
| `convergence_factor` | `convergence_factor.py::convergence_factors` of the system's `convergence` rows; null without a stored history, without an iteration in it, or for a zero initial residual |
| `true_residual_rel` | `solve_attempts.true_residual_rel`; null when unmeasured |
| `stopping_threshold_rel` | `true_residual.stopping_threshold_rel(rtol, atol, rhs_norm)` from `planned_systems` and `solve_attempts.rhs_norm`; null without a nonzero measured RHS norm |
| `timing_spread` | `(max - min) / median` of `solve_attempts.samples_s`; 0 for one sample, null without a solve row or for a zero median |

Vocabulary. A **timing repeat** is the same solve timed again to cut noise; it is already reduced
inside `solve_s` and is never an element of a resample. A **bootstrap resample** is an in-memory
redraw of whole matrices. An **RHS** is one system of a matrix. **Allocated** or **amortized
online** time shares a matrix setup among its planned RHS; **standalone** time charges every RHS
the whole setup. `mean_over_systems` pools all selected systems; `mean_over_matrices` averages
the per-matrix means.

Notation. Method $p$; matrix $m \in \mathcal{M}$, $K = |\mathcal{M}|$; system $s$ = (matrix, RHS), $m(s)$ its matrix; $r_m$ the planned RHS count of $m$; $\mathcal{A}_p$ attempted systems, $\mathcal{C}_p \subseteq \mathcal{A}_p$ converged systems of $p$; $k_{p,s}$ iterations. $\tau_{p,s,j}$ is the wall time of **timing repeat** $j = 1..R$ of the same solve; $\operatorname{stat}$ is the configured `TimingStatistic` (median default).

$$t^{\mathrm{solve}}_{p,s} = \operatorname{stat}_{j}\,\tau^{\mathrm{solve}}_{p,s,j}, \quad t^{\mathrm{setup}}_{p,m} = \operatorname{stat}_{j}\,\tau^{\mathrm{setup}}_{p,m,j}$$

$$t^{\mathrm{online}}_{p,s} = t^{\mathrm{solve}}_{p,s} + \frac{t^{\mathrm{setup}}_{p,m(s)}}{r_{m(s)}}\ \text{(amortized: setup shared by the matrix's planned RHS)}, \qquad t^{\mathrm{standalone}}_{p,s} = t^{\mathrm{solve}}_{p,s} + t^{\mathrm{setup}}_{p,m(s)}\ \text{(every RHS pays the whole setup)}$$

$$T^{\mathrm{solve}}_p = \sum_{s \in \mathcal{A}_p} t^{\mathrm{solve}}_{p,s}, \qquad T^{\mathrm{online}}_p = \sum_{m} t^{\mathrm{setup}}_{p,m} + T^{\mathrm{solve}}_p \quad\text{(workload totals; failed attempts keep their time)}$$

A mean of $t^{\mathrm{online}}$ over converged systems is **not** $T^{\mathrm{online}}_p/|\mathcal{S}|$: a failed system's setup share is simply missing.

**Populations** for a metric $x$ defined on a set $\mathcal{D}_x$ of outcomes: $\mathcal{P}^{\mathrm{own}}_p = \{ s : \mathrm{outcome}_{p,s} \in \mathcal{D}_x \}$ and $\mathcal{P}^{\mathrm{common}} = \bigcap_{p'} \mathcal{P}^{\mathrm{own}}_{p'}$; $\mathcal{P}_{p,m}$ = its systems on matrix $m$, $\mathcal{M}_{\mathcal{P}} = \{ m : \mathcal{P}_{p,m} \neq \emptyset \}$. `Population.OWN` and `Population.COMMON` select them.

**Two means, never called just "mean":**

$$\bar{x}^{\mathrm{sys}}_p = \frac{1}{|\mathcal{P}_p|} \sum_{s \in \mathcal{P}_p} x_{p,s}\ \text{(crude pooled)}, \qquad \bar{x}^{\mathrm{mat}}_p = \frac{1}{|\mathcal{M}_{\mathcal{P}}|} \sum_{m \in \mathcal{M}_{\mathcal{P}}} \frac{1}{|\mathcal{P}_{p,m}|} \sum_{s \in \mathcal{P}_{p,m}} x_{p,s}\ \text{(every matrix weighs the same)}$$

**Quantiles** (linear interpolation, the numpy/polars default, which is also what seaborn's violin box draws), $h = (n-1)\alpha + 1$: $q_\alpha = x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)(x_{(\lfloor h \rfloor + 1)} - x_{(\lfloor h \rfloor)})$. Box = $q_{0.25}, q_{0.5}, q_{0.75}$; whiskers $q_{0.05}, q_{0.95}$.

**Matrix-cluster bootstrap interval** (reads stored values only; $B$ = `BOOTSTRAP_RESAMPLES`, $\alpha = 1 -$ `CONFIDENCE_LEVEL`). For resample $b = 1..B$ draw $m^{(b)}_1..m^{(b)}_{K'}$ uniformly with replacement from $\mathcal{M}_{\mathcal{P}}$, $K' = |\mathcal{M}_{\mathcal{P}}|$, and recompute the bar's statistic:

$$\bar{x}^{\mathrm{sys},(b)} = \frac{\sum_{i}\sum_{s \in \mathcal{P}_{p,m^{(b)}_i}} x_{p,s}}{\sum_{i} |\mathcal{P}_{p,m^{(b)}_i}|}, \quad \bar{x}^{\mathrm{mat},(b)} = \frac{1}{K'}\sum_{i} \frac{1}{|\mathcal{P}_{p,m^{(b)}_i}|}\sum_{s \in \mathcal{P}_{p,m^{(b)}_i}} x_{p,s}, \quad \mathrm{CI} = [q_{\alpha/2}, q_{1-\alpha/2}]\ \text{of}\ \{\bar{x}^{(b)}\}_{b=1}^{B}$$

Seeded with `DEFAULT_RANDOM_SEED`; `None` if $K' < 2$; a matrix with no system in the population is dropped. Nothing about the solves is recomputed: it resamples stored per-system values, grouped by matrix, because RHS of one matrix share it and the effective sample size is $K$. It covers matrix-to-matrix variability only, not timing noise. `benchmarking/uncertainty.py::matrix_cluster_interval` computes it for `ClusterStatistic.MEAN_OVER_SYSTEMS`, `MEAN_OVER_MATRICES` and `MEDIAN`.

**Time per iteration** (per system with a real iteration count $k \ge 1$; `UNCONVERGED` runs count because time and iterations are both real and a rate is not censored; `SOLVER_ERROR` and setup failures do not): $\theta_{p,s} = t^{\mathrm{solve}}_{p,s} / k_{p,s}$. It isolates per-iteration cost (preconditioner application, flexible-CG overhead) from numerical effectiveness; fixed per-solve overhead inflates $\theta$ when $k$ is small, which is why a figure's bar is the ratio of sums $\Theta_p$ (`time_per_iteration_summary`, below) and the per-system $\theta_{p,s}$ are drawn only as points.

**Convergence factor** (per converged system with a stored residual history): the stored `convergence` frame holds the relative residual norm $\hat{r}_{p,s,j} = \|r_j\| / \|b\|$ as the solver recorded it during the solve (`residual_rel`) at iterations $j = 0..J_{p,s}$, $j = 0$ the initial residual and $J_{p,s}$ the last stored iteration. $\gamma_{p,s} = (\hat{r}_{p,s,J} / \hat{r}_{p,s,0})^{1/J_{p,s}}$ is the geometric mean of the per-iteration reductions, so $\gamma < 1$ and smaller is faster; it is the same however the reduction was spread over the iterations, and a zero final residual gives 0. It is undefined for $J = 0$ and for a zero initial residual (a zero right-hand side). Only `CONVERGED` systems enter: an `UNCONVERGED` history stopped where the cap or a breakdown cut it, so its factor does not describe a solve that reached the tolerance, and the figure states $k/n$ so the excluded systems stay visible. The history is read in memory from `convergence.parquet` into the observation view; no stored schema changes.

`reporting/metrics.py::METRICS` is the catalog; its labels and formulas are the single source and this table repeats them verbatim (`tests/architecture/test_metric_catalog_documented.py` fails on any difference). The failure rule is the `defined_on` column: a system whose outcome is outside it has no value for the metric and is excluded from every mean, quantile and interval, never counted as zero and never given the iteration cap.

| `MetricId` | Label | Observation unit | Defined on | Formula |
|---|---|---|---|---|
| `ITERATIONS` | iterations to converge | system | `converged` | $k_{p,s},\quad s \in \mathcal{C}_p$ |
| `SOLVE_S` | solve time per system [s] | system | `converged` | $t^{\mathrm{solve}}_{p,s} = \operatorname{stat}_{j}\,\tau^{\mathrm{solve}}_{p,s,j}$ |
| `ONLINE_S` | online time per system [s] | system | `converged` | $t^{\mathrm{online}}_{p,s} = t^{\mathrm{solve}}_{p,s} + \frac{t^{\mathrm{setup}}_{p,m(s)}}{r_{m(s)}}$ |
| `STANDALONE_S` | standalone time per system [s] | system | `converged` | $t^{\mathrm{standalone}}_{p,s} = t^{\mathrm{solve}}_{p,s} + t^{\mathrm{setup}}_{p,m(s)}$ |
| `SETUP_S` | setup time per matrix [s] | matrix (one value per method, dataset, matrix) | `converged`, `unconverged`, `solver_error` (the setup completed) | $t^{\mathrm{setup}}_{p,m} = \operatorname{stat}_{j}\,\tau^{\mathrm{setup}}_{p,m,j}$ |
| `TIME_PER_ITERATION_S` | time per iteration [s] | system | `converged`, `unconverged` | $\theta_{p,s} = \frac{t^{\mathrm{solve}}_{p,s}}{k_{p,s}}$ |
| `CONVERGENCE_FACTOR` | residual reduction factor per iteration | system | `converged` | $\gamma_{p,s} = \left(\frac{\hat{r}_{p,s,J}}{\hat{r}_{p,s,0}}\right)^{1/J_{p,s}},\quad s \in \mathcal{C}_p$ |

`reporting/summary.py` applies the rule in one place. `select(obs, metric, population)` keeps the
rows on which the metric is defined (one row per matrix for a matrix metric); `Population.COMMON`
keeps only systems, or matrices, defined for every method present in `obs`, so a method with no
defined unit empties it. `summarize(obs, metric, population)` gives one `SUMMARY_SCHEMA` row per
method: `n_planned` and `n_defined` count the metric's observation units, `n_matrices` the matrices
with a defined unit, and `n_unconverged`, `n_solver_error`, `n_setup_failed`, `n_not_attempted`
always count systems, whichever the metric's unit. A method with no defined unit has
`n_defined = 0` and null statistics, never 0. `mean_over_systems_ci_*` and
`mean_over_matrices_ci_*` are the 95% matrix-cluster bootstrap interval, null below two matrices.
`time_per_iteration_summary(obs, population)` gives one `RATIO_OF_SUMS_SCHEMA` row per method
(`n_planned`, `n_defined`, `n_matrices` as above, `solve_s_sum`, `iterations_sum`,
`ratio_of_sums`) over the systems `select` keeps for `TIME_PER_ITERATION_S`:
$\Theta_p = \sum_{s \in \mathcal{P}_p} t^{\mathrm{solve}}_{p,s} / \sum_{s \in \mathcal{P}_p} k_{p,s}$,
which weighs every iteration the same, unlike the mean of $\theta_{p,s}$ that lets many short solves
(whose fixed per-solve overhead inflates their rate) outvote a few long ones; the sums and ratio are
null when the method has no such system.
`reporting/cohorts.py::by_cohort(obs, statistic, rows, schema=...)` splits any per-method statistic
by RHS recipe: one block of `statistic`'s rows per recipe of `obs` (in order of first appearance,
each computed on that recipe's systems only, so `Population.COMMON` is taken within the recipe),
behind a leading `cohort` column holding the recipe's lossless key; with
`CohortRows.RECIPES_AND_MIXTURE` a last block is `statistic` of the whole of `obs`, labelled
`MIXTURE_COHORT` (`"mixture"`, never a recipe key). `COHORT_SUMMARY_SCHEMA` and
`COHORT_RATIO_OF_SUMS_SCHEMA` are `cohort_schema` of `SUMMARY_SCHEMA` and `RATIO_OF_SUMS_SCHEMA`.
`reporting/matrix_breakdown.py::matrix_breakdown(obs, metric)` keeps each matrix apart: one
`MATRIX_BREAKDOWN_SCHEMA` row per method and planned matrix (`method`, `metric`, `dataset`,
`matrix_id`, `n_planned` planned RHS of the matrix, `n_defined` RHS that define the metric, `mean`),
sorted by method, dataset and matrix, with
$\bar{x}_{p,m} = \frac{1}{|\mathcal{P}^{\mathrm{own}}_{p,m}|} \sum_{s \in \mathcal{P}^{\mathrm{own}}_{p,m}} x_{p,s}$
over the method's own population on the matrix; `mean` is null, never 0, when no RHS of the matrix
defines the metric. A common population would empty every cell of a matrix on which any method
failed, which is what the breakdown is meant to show. A per-matrix metric (`SETUP_S`) raises
`ValueError`, since it has no RHS to average over.
`residual_consistency(obs)` counts, per method, converged systems whose true residual exceeds
`RESIDUAL_MISMATCH_FACTOR` times the stopping level and reports the worst ratio; classification
is never changed by it. `timing_noise(obs)` gives the median `timing_spread` per method and
`outcome_counts(obs)` the number of planned systems per method in each `Outcome`
(`OUTCOME_COUNTS_SCHEMA`); figure cards record both. `benchmarking/uncertainty.py::MatrixClusterBootstrap`
(`matrices`, `level`, `resamples`, the last two defaulting to `CONFIDENCE_LEVEL` and
`BOOTSTRAP_RESAMPLES`, the values `matrix_cluster_interval` resamples with) is what a card states
about an interval it draws.

Reproduce a table from the stored parquets of a session:

```python
import polars as pl

from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.observations import MethodFrames, observations_of_methods
from neuralls.domain.solver.reporting.summary import Population, summarize
from neuralls.platform.reporting.benchmark_events import (
    read_convergence,
    read_setup_events,
    read_solve_attempts,
)

frames = {
    method: MethodFrames(
        planned_systems=pl.read_parquet(directory / "planned_systems.parquet"),
        setup_events=read_setup_events(directory / "setup_events.parquet"),
        solve_attempts=read_solve_attempts(directory / "solve_attempts.parquet"),
        convergence=read_convergence(directory / "convergence.parquet"),
    )
    for method, directory in method_directories.items()
}
table = summarize(observations_of_methods(frames), MetricId.ONLINE_S, Population.COMMON)
```

## Benchmarking statistics

### Comparison vocabulary

The benchmark separates costs by when they are paid. **Generation** is the
offline construction of a dataset or training artifact; it is reported only
when a measured historical value is available. Checkpoint and input-data
loading, validation, staging, plotting, persistence, and MLflow work are
excluded from online timing. **Setup** is method preparation for one matrix;
**solve** is one attempted RHS. **Workload online time** is the sum of setup
and solve events for the workload. **Amortized online time per RHS** is solve
time plus that matrix setup divided by the planned RHS count. Thus no
checkpoint or data-read cost is silently charged to a method.

The **success rate** is converged attempts divided by planned RHS systems;
failures remain in the denominator and are never assigned a penalty time. An
**empirical cumulative distribution function (ECDF)** shows the fraction of
planned systems solved within an absolute time budget. A **Dolan--Moré
performance profile** instead shows the fraction within a factor of the
fastest method on each system. Profiles are emitted only for one homogeneous
recipe; heterogeneous recipes are reported as workload mixtures.

Every planned system ends in exactly one **outcome** (`reporting/observations.py::Outcome`).
`solve_outcome` / `solve_outcome_expr` derive it from the stored `solve_status` and
`residual_status` columns, never from an iteration count (zero iterations is a valid count for a
solved zero right-hand side and for `max_iterations = 0`). `validate_decoded_solve_frame` rejects
rows that contradict the derivation: a failed solve with an unmeasured residual must store 0
iterations, and a measured residual needs a nonzero right-hand side.

| Outcome | What happened | Stored | Measured time kept | Iteration count |
|---|---|---|---|---|
| `CONVERGED` | tolerance reached (a zero right-hand side converges unmeasured) | `solve_status = converged` | solve time | real |
| `UNCONVERGED` | the solver returned an iterate that missed the tolerance (iteration cap reached, or a solver-flagged breakdown that still returned a finite iterate; breakdown is not stored) | `failed` + `residual_status = measured` | full solve time | real: where the solver stopped; when the cap was hit, a lower bound on the iterations needed |
| `SOLVER_ERROR` | the solver raised, or returned a non-finite iterate (or one whose true residual is non-finite) | `failed` + `residual_status = unmeasured` | time until the error, or the full solve time for a non-finite iterate | none (stored as 0, not a count); a non-finite iterate keeps the real count in the failure reason (`non-finite iterate after 37 iterations`) |
| `SETUP_FAILED` | matrix setup raised; no right-hand side of that matrix ran | failed setup event, no solve row | setup time until the error | none |
| `NOT_ATTEMPTED` | planned, never run | planned system only, no solve row, no failed setup | none | none |

`SETUP_FAILED` and `NOT_ATTEMPTED` have no solve row, so the join of planned systems with setup
events decides them; the solve-row derivation covers the first three.

A **paired time** comparison uses the same matrix/RHS system for a configured
reference and method. Its **speed ratio** is reference time divided by method
time, so values above one favor the method. Pair counts retain both-solved,
reference-only, and method-only failures. Uncertainty resamples whole
**(dataset, matrix) clusters**, preserving dependence among sibling RHS
observations.
The reference is always the explicitly configured
`reference_preconditioner`; no observed result may select a fallback. If the
reference is absent or unavailable, paired ratios are omitted.

These are deterministic heterogeneous systems, not repeated COCO-style
restarts of one stochastic problem, so ordinary benchmark sweeps do not have
a meaningful expected running time (ERT). ERT would be appropriate only for
a separately designed restart experiment with a specified success
distribution and stopping rule.

`domain/benchmarking/` provides, all over the observation frame as polars frames:

- Dolan–Moré performance profiles (`performance_series`) and absolute ECDFs (`ecdf_series`) that
  retain failures in their denominators (`profiles.py`);
- paired reference/method systems (`paired_systems`), their per-system ratios (`speed_ratios`),
  the speed-ratio summary using only positive-value both-solved systems while retaining all
  both-solved, reference-only, and method-only counts (`speed_ratio_summary`), and `break_even`
  (`paired.py`), with its cumulative-cost curves (`break_even_curves`, `cost_curves.py`);
- `reference_among_methods`, which turns the configured `ReferenceSelection` into the reference
  name a group can be compared to;
- percentile intervals that resample whole `(dataset, matrix_id)` clusters with
  every sibling RHS observation (`uncertainty.py`); paired Wilcoxon p-values are not reported;
- the reference selection types (`selection.py`).

They are described under "Profiles, paired comparisons and break-even over the observation view".

Performance profiles normalize by the fastest method on each problem. A
failed attempt remains censored rather than receiving a penalty time, and
all-failed systems remain in every profile denominator. Paired
comparisons require a `ConfiguredReference`; `NoReferenceConfigured` and
`ReferenceUnavailable` produce no ratios, and no observed outcome can select a
fallback reference.

### Profiles, paired comparisons and break-even over the observation view

These functions read the observation frame and take a `MetricId` with one value per system
(`ITERATIONS`, `SOLVE_S`, `ONLINE_S`, `STANDALONE_S`, `TIME_PER_ITERATION_S`, `CONVERGENCE_FACTOR`); a per-matrix metric
raises `ValueError`. The failure rule is `select`'s, so "solved" means the metric is defined on the
system's outcome.

`ecdf_series(obs, metric)` and `performance_series(obs, metric)` return a `ProfileSeries`:
`frame` has `PROFILE_SERIES_SCHEMA` (`method, x, fraction, n_planned, n_solved`), one row per solved
system, ascending in `x`, with `fraction` the number of solved systems of the method with value at
most `x` divided by `n_planned`, so a curve ends at `n_solved / n_planned` and ties share the
fraction of the whole tie. `planned` has `PLANNED_SCHEMA` (`method, n_planned, n_solved`), one row
per method of `obs`: a method that solved nothing has no curve row and is still counted there.
The ECDF `x` is the metric value (absolute budget, independent of the other methods). The
performance-profile `x` is the Dolan-More ratio $\rho_{p,s} = x_{p,s} / \min_{p'} x_{p',s}$ with
the minimum over the methods that solved $s$; an unsolved system stays in the denominator.

A system whose best value is 0 (a solve at the initial guess takes 0 iterations, a time can be 0)
has no ratio for any method. It is dropped from numerator and denominator of every method and
counted in `ProfileSeries.n_excluded_zero_best`; leaving it in the denominators would lower every
method's plateau by a system nobody failed on, which reads as failures. The ECDF keeps such a
system, since an absolute budget of 0 is meaningful.

`paired_systems(obs, reference, metric)` has `PAIRED_SCHEMA` (`method, system_id, dataset,
matrix_id, pair_kind, reference_value, method_value`): for each non-reference method, one row per
system planned for both where at least one side has a value. `pair_kind` is a `PairKind`: `both`,
`method_only` (no `reference_value`) or `reference_only` (no `method_value`). A system neither side
solved has no row. `speed_ratio_summary(obs, reference, metric, seed=...)` has
`SPEED_RATIO_SCHEMA`, one row per non-reference method:

| Column | Meaning |
|---|---|
| `n_systems_both_solved` | `both` rows, including those with a zero value |
| `n_ratio` | `both` rows with both values positive, the systems a ratio $x_{\mathrm{ref},s} / x_{p,s}$ exists for; a zero value makes it undefined, never infinite |
| `n_matrices` | distinct `(dataset, matrix_id)` among the ratio rows: the cluster count the interval rests on, since sibling RHS share a matrix and `n_ratio` overstates the effective sample size |
| `n_reference_only`, `n_method_only` | one-sided systems |
| `median_ratio`, `ci_low`, `ci_high` | median ratio (above 1 favours the method) and its matrix-cluster bootstrap interval; null without a ratio, interval null below two matrices |

`speed_ratios(paired)` lists the per-system ratios behind the summary (`method`, `dataset`,
`matrix_id`, `system_id`, `ratio`), so a ratio CDF and the summary count the same systems. Reports
compare methods to the configured reference on `REFERENCE_METRIC` (`ONLINE_S`, amortized online
time); `reference_among_methods(obs, selection)` gives the reference name only for a
`ConfiguredReference` whose method has systems in the group.

`break_even(obs, reference)` has `BREAK_EVEN_SCHEMA` (`method, setup_s_mean,
reference_setup_s_mean, solve_s_mean, reference_solve_s_mean, break_even_rhs`). The means are over
the population common to the pair alone: `SETUP_S` over the matrices both set up, `SOLVE_S` over
the systems both solved. Against the reference, on those populations,
$C_p(n) = \bar{t}^{\mathrm{setup}}_p + n\,\bar{t}^{\mathrm{solve},\mathrm{sys}}_p$ and

$$n^{*} = \frac{\bar{t}^{\mathrm{setup}}_p - \bar{t}^{\mathrm{setup}}_{\mathrm{ref}}}{\bar{t}^{\mathrm{solve},\mathrm{sys}}_{\mathrm{ref}} - \bar{t}^{\mathrm{solve},\mathrm{sys}}_p}$$

defined only when the denominator is positive; otherwise `break_even_rhs` is null (the method is
not faster per RHS and never breaks even). With a positive denominator and a setup no dearer than
the reference's, `break_even_rhs` is 0.0: the method never costs more. A pair with no common unit
has null means and a null `break_even_rhs`.

`cost_curves.py::break_even_curves(obs, reference)` samples the two lines whose crossing is $n^*$,
so a break-even figure draws exactly what its table stores. It has `COST_CURVE_SCHEMA` (`method,
side, n_rhs, cost_s, break_even, break_even_rhs, break_even_cost_s`): one row per compared method,
`side` (`CostSide`: `method` or `reference`) and $n = 1..N$, with $C(n)$ in `cost_s` from that
pair's `break_even` means. $N$ is the largest planned RHS count of a matrix (`planned_rhs`),
extended to $\lceil n^* \rceil$ of the furthest break-even count but not beyond
`BREAK_EVEN_HORIZON_LIMIT` (1000), since a near-zero per-RHS saving puts $n^*$ arbitrarily far
out. `break_even` holds a `BreakEvenState` repeated on every row of a method: `breaks_even`
(`break_even_rhs` is $n^*$ and `break_even_cost_s` $C_p(n^*)$), `never` (not faster per RHS; both
null) or `no_common_systems` (a mean is null, so `cost_s` is null too).

## Diagnostic solver results

`CGComparisonResult` remains a solver-diagnostic DTO. Its
`solve_time_seconds` is the explicit PCG/FCG measurement and
`offline_generation_time_seconds` exposes available historical generation
cost separately. It does not aggregate generation, setup, and solve into an
online total; setup-aware aggregate reporting derives totals only from linked
benchmark events.

Per-iteration residual and energy-error histories, timing samples, and memory
diagnostics remain available for artifacts and single-run plots. Workload
ranking uses complete event summaries.

Three different residual numbers exist and none replaces another. The solver's
stopping test and `SolveConverged.final_relative_residual` use its recursive estimate.
`SolveAttempt.residual` (`MeasuredResidual`/`UnmeasuredResidual`) is `||b - A x||/||b||`
recomputed from the returned solution after the timed region (`true_residual.measure_residual`);
it never changes how a solve is classified, and it is unmeasured when the solver raised or the
right-hand side is zero. `measure_residual` returns `NonFiniteResidual` for a non-finite iterate or
a residual that overflows; `run_cg_comparison` then records the solve as a failed one with an
unmeasured residual, no histories and no energy-error second solve, instead of storing a NaN. `SolveAttempt.exact_reference` (`ExactSolutionSolved` with the
reference solution's achieved residual, or `ExactSolutionUnavailable` with the reason) records
the Jacobi-PCG solve that supplies `x_exact`. An exact energy-error history exists only when
that reference solved, and it always comes from the second, untimed diagnostic solve, because
the timed solve never receives `x_exact`. `true_residual.stopping_threshold_rel` gives the level
at which the solver stops (`max(rtol, atol/||b||)`, taken from torchalg's own criterion).

## Presentation planning

`GroupPresentation` lives in `models/group_presentation.py` (it shares no name with the shared presentation modules) and carries a typed `scope` (`models/figure_scope.py`: `RecipeScope | DatasetScope | AllDatasetsScope`, plus `MethodScope` for method figures) that sinks render; a `RecipeScope`, a `SystemTag`, a `SolveAttempt` and the report labels hold the typed `RhsRecipe` (`shared/rhs_recipe.py`), whose machine key is the event frames' `recipe` column and the recipe segment of a group id; `PreconditionerPlotStyle` carries the method's `PreparedMethodPresentation` with family and colour/marker keys, never text.

Presentation planning (`prepare_presentations`, `reduce_realized_facts`) lives in `domain/presentation/planner.py` (see its README). `GroupPresentation.methods` carries a `GroupMethodPresentation` for each method that has systems in that group: its prepared presentation plus the realized facts invariant over the method's systems in the group (only facts that `is_compact_by_policy` accepts join the compact view). A realized fact is invariant only if every system of the method in the group was realized and they agree, so a group containing a system whose setup failed states none.

## Boundary rule

`BenchmarkMethodSink` is the narrow persistence port for a completed method
workload. Its domain-owned `MethodWorkload` payload carries the validated
method configuration and diagnostics as required data alongside canonical
events; the port
accepts only `(workload, summary)`. Metric calls are forbidden during event
emission. MLflow run ownership stays in composition, so the domain port cannot
create system runs or method grandchildren.

Composition maps typed configuration and loaded tensors to `torchalg` runtime
objects. Platform code persists and plots returned domain payloads. Domain code
does not parse TOML, open MLflow runs, resolve filesystem layout, or write
artifacts.

## Stage costs

`models/cost.py::StageCost` is a closed sum for a method's offline (dataset-generation) cost:
`NotApplicable` (the method has no training data; reads as zero in totals and plots),
`Known(wall_time_seconds, peak_memory_bytes, provenance)` (`MEASURED` or `HISTORICAL`), and
`Unavailable(reason)` (the cost applies but was never recorded; it carries no number, so it is never
read as zero). Only `composition/comparison/_generation_cost.py::resolve_generation_cost` chooses the
variant. `recorded_seconds` returns the seconds of a `Known` cost and `None` otherwise.

## Timing-repeat agreement

`shared/device.py::repeat_timed_outcome` repeats a timed region per the `TimingRepeatPolicy` and
returns a `RepeatAgreement` with the timing: `SingleRun` (one timed execution), `Unchecked`
(several, but no fingerprint to compare), `Agreed`, or `Diverged(first, other)` (a repeat's
iteration count differed from the first, so its samples are not comparable; the first divergence is
logged once). `SolveAttempt.repeats` stores it for every solve; a solve that raised is `Unchecked`, a non-finite iterate keeps the agreement of its completed repeats.

## Split records

`models/split_record.py::DatasetSplitRecord` states what a dataset's pinned split held out: the split digest, fold row counts and the matrix ids of the fit (train plus validation) and test folds. `build_split_record` rejects a row in both a fit fold and the test fold and a row outside the dataset. A single-matrix dataset is `SingleMatrixShared`, because its one matrix is in every fold by design and only RHS rows are held out; a multi-matrix dataset must be `MatrixDisjoint`, so a matrix in both folds is rejected on construction.
