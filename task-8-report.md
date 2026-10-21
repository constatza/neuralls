# Task 8 report: typed recipe profiles and workload mixtures

## Outcome

Report classification now has two explicit variants:

- `RecipeProfileReport` is fine-level and contains one ordered recipe. Its
  profile and ECDF plots remain recipe-specific.
- `WorkloadMixtureReport` is medium/global and preserves complete ordered
  recipe membership. It emits aggregate convergence bars but no pooled profile
  or ECDF data.

Both variants retain matrix and planned RHS counts, while each method's
convergence summary retains planned, attempted, and converged counts. The
composition report path dispatches on the closed variants, not rendered label
strings.

## RED/GREEN evidence

- RED characterization added for mixed two-recipe groups and sink dispatch;
  the new behavior was absent before the variant/sink implementation.
- GREEN: focused classifier, composition, and sink suite passes with 13 tests.
- Targeted `ty` check passes for domain, composition, outcome plots, and
  tracking.
- Targeted Ruff lint/format, `tach check`, and `git diff --check` pass.

## Scope boundary

This task only controls report typing and plot scheduling. Figure titles,
manifest metadata, and subsequent plotting redesign remain later work.
