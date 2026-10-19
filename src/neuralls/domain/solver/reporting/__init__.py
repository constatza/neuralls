"""Long-form metrics table over comparison results, and its three report levels.

Every number a report shows is "this preconditioner, over these systems". The
preconditioner is therefore the defining key of every level and never a level
of its own; a level only chooses which *other* columns also key the groups.
Missing measurements are nulls, which polars skips in means and spreads, so an
unmeasured solve is never silently read as 0.0.

Metrics and references. A run stopped by the iteration cap is a failure, i.e. a
right-censored observation, and is never averaged in as if it were a solve time
and never given an invented penalty:

- Success rate: converged / run, shown with every number, because a statistic
  over converged runs alone says nothing about the runs that failed (Gould &
  Scott 2016, "A note on performance profiles for benchmarking software", ACM
  Trans. Math. Softw. 43, on reading summaries beside raw outcomes).
- Projection means (solve time, iterations): over all runs, failed runs at their
  measured cost, so the projected cost per solved system is the pooled ERT. Their
  spreads (std) stay over converged runs, the spread of a successful solve.
- ERT, expected running time: one-time shared cost plus the recurring time of all
  runs, failed included, divided by the number of successes, a pooled cost per solved system and not the expectation of
  a restart strategy (Hansen et al. 2016, "COCO: Performance Assessment",
  arXiv:1605.03560). Its interval is a bootstrap over systems.
- Recurring time: `recurring_s` is a run's time to solution minus the setup and
  generation its sharing policy flags as one-time (a checkpoint load copied into
  every sibling by the setup cache). Profiles, ECDF, scatters and speedups use
  it, so a shared cost is not charged once per system; ERT, the projection and
  break-even charge it once. A setup that depends on each system's matrix is a
  per-system cost and stays in `recurring_s`.
- Iteration statistics (mean, median, std) over converged runs only.
- Performance profiles (Dolan & Moré 2002, "Benchmarking optimization software
  with performance profiles", Math. Program. 91) and ECDF profiles (Hansen et
  al. 2016), which keep failures as ratio infinity or as unsolved, are built from
  `outcomes_by_system` by `domain/benchmarking/profiles.py`.
"""
