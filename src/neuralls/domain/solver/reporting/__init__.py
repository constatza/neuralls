"""Long-form metrics table over comparison results, and its three report levels.

Every number a report shows is "this preconditioner, over these systems". The
preconditioner is therefore the defining key of every level and never a level
of its own; a level only chooses which *other* columns also key the groups.
Missing measurements are nulls, which polars skips in means and spreads, so an
unmeasured solve is never silently read as 0.0.
"""
