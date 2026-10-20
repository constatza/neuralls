# Task 6 report: method-owned MLflow tracking

## Outcome

The batch now bypasses legacy per-system comparison reuse and gates reuse only
on complete schema-v1 method children. The method gate runs before opening the
session parent, honors `force`, and avoids an empty parent when every method is
reusable. A method child owns canonical setup/solve artifacts, validated
summary/config artifacts, residual and energy histories, diagnostics, and
method figures; the session owns cross-method reports. No system runs or method
grandchildren are created.

Method identity uses the Task 5 benchmark schema version digest, the resolved
method config/checkpoint identity digest, and complete planned membership. A
candidate is reused only after downloading and validating every declared raw
artifact, Task 5 Parquet schemas/setup links, and summary method/config/
membership identity. The parsed method config must equal the expected
canonical config with unset fields omitted; diagnostic JSON membership and
method identity, event-frame methods, and summary counts/totals are reconciled
with the planned workload. Legacy, corrupt, wrong-schema, incomplete, or
config-less children miss.

The domain sink is the narrow `record_method(workload, summary)` port. Its
domain-owned workload payload requires a typed method configuration and
diagnostic records, so the platform sink consumes one complete contract rather
than strengthening it with a second platform protocol. Generation remains the
only optional offline cost; checkpoint and data I/O are not metrics.

The persisted method configuration is the shared canonical representation:
checkpoint inputs are content digests and cosmetic/runtime lease paths are
omitted. This keeps artifact comparison aligned with the existing
method-config/checkpoint identity, so equal checkpoint bytes leased at
different local paths reuse the same method child while changed bytes miss.

## TDD and verification

- RED: default AMG and checkpoint-backed config tests failed on unset optional
  TOML fields; recursive canonical omission fixed both cases.
- RED: corrupt diagnostics and wrong method-config tests failed until reuse
  downloaded and validated every required artifact's contents.
- RED: the two-lease checkpoint regression failed until persisted config used
  content identity instead of the resolved local checkpoint path; it now passes
  for equal bytes and misses for changed bytes.
- Focused sink, artifact, topology, lifecycle, batch-scoping, comparison, and
  restored plotting tests pass.
- SQLite integration covers valid method reuse without a new session, forced
  execution with a new session, child-only histories/figures, and no method
  grandchildren.

## Scope

Task 9 plotting changes were removed: parent reports retain the pre-existing
system scatter, logarithmic paired plot, speedup naming, and reference
selection behavior. The report is stored in this SDD workspace rather than the
repository root.
