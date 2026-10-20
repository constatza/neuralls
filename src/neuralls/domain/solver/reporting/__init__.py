"""Pure reporting reductions over canonical benchmark events.

Setup events and solve attempts remain separate. ``event_frames`` normalizes
their closed domain variants at the Polars boundary; ``aggregation`` validates
links, allocates each matrix setup across its complete planned RHS bundle, and
derives solve-only and amortized-online outcomes. Failures remain censored in
profiles while their measured attempt time stays in workload totals.

Report-level summaries retain planned, attempted, converged, and matrix counts
independently. Confidence intervals resample whole matrix clusters so sibling
RHS observations are never treated as independent matrices.
"""
