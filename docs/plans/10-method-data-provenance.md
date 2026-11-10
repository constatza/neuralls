# 10: Separate method training data from comparison matrix sets

## Goal

Make a method's data dependencies explicit before comparison execution, so a bare method is never described as trained on a dataset, a checkpoint is never attributed to an unverified fit dataset, and a comparison matrix set remains the scope on which *every* method is measured. Reuse the existing presentation and report machinery; do not introduce a new preconditioner framework.

## Current workflow and the distinction

```text
case config
  ├─ comparison matrix set → pinned test fold → (matrix A, RHS recipe) systems
  └─ method specs
       ├─ identity / Jacobi / IC0 / geometric AMG → matrix A only
       ├─ POD-2G inline fit → snapshot dataset → fit at matrix setup
       ├─ checkpoint-backed POD-2G → checkpoint (basis already fitted)
       ├─ neural preconditioner → checkpoint; optional extra arrays from the comparison system
       └─ neural POD-2G → checkpoint + parameter dataset used at setup
method × comparison matrix → one timed setup → sibling RHS solves → events
events + method presentation + comparison scope → MLflow runs, logs and report figures
```

The **training/fit dataset** is a dependency of some methods. The **comparison matrix set** supplies `A` and possibly a dataset RHS to every method, including Jacobi. An event row `(Jacobi, comparison matrix set, matrix)` is correct; a legend or method-run name `Jacobi | <training dataset>` is not. Titles, scope run names and logs must say that a method was *evaluated on* the comparison matrix set without making that set part of the method's training identity. `rhs_source = "dataset"` selects held-out RHS rows from the comparison matrix set; other RHS recipes still use its matrices.

Today `StandaloneMethodPresentation | DatasetQualifiedMethodPresentation` already prevents a Jacobi fit-dataset qualifier in the comparison method renderer (`presentation_preflight.py`, `domain/presentation/render.py`). Method run names and legends use that renderer; figure scopes use a separate matrix-set presentation. Keep that separation. The gaps are (1) checkpoint-backed POD-2G still requires `coarsening.dataset_dir` and uses it for the method label and generation cost without proving it is the checkpoint's fit dataset (`_offline_cost.py` documents this), (2) a comparison neural method has no verified training-dataset presentation even though its training run does, and (3) generic `dataset` names in logs and stored events can make the two roles hard to read.

## Design

Use a closed sum at the **resolved comparison-method seam**, after config and checkpoint resolution:

```text
MethodDataOrigin =
    NoFitData
  | InlineFit(fit_dataset)
  | CheckpointBased(checkpoint_refs, training_data)

CheckpointTrainingProvenance = VerifiedTrainingDataset(dataset, digest)
                             | TrainingDataUnavailable(reason)
```

`NoFitData` is the no-data case for Identity, Jacobi, IC0 and geometric AMG; it belongs to the outer sum, not the checkpoint-only provenance union. `checkpoint_refs` is a nonempty tuple of the already resolved refs, not a second copy of checkpoint paths. `fit_dataset` is the dataset whose snapshots the inline fit actually reads. The checkpoint variant may have no verifiable training-data record, including an explicit local checkpoint; that is `TrainingDataUnavailable`, never an invented dataset. A method with several checkpoints can be displayed with one training dataset only when all refs attest to the same digest; otherwise keep the provenance as unavailable for a single-dataset label and retain each checkpoint's existing identity/cost handling. `NeuralPODCoarseningConfig.dataset_dir` is a **runtime parameter source**, not proof of the network's training dataset; it remains separate from `MethodDataOrigin`. Likewise, neural `extra_input_names` name comparison-system inputs, not training inputs.

Construct this value once from the resolved config, existing checkpoint resolution metadata, training-run identity tags, and dataset manifests. Do not add a bool such as `uses_dataset` or a nullable field whose meaning depends on method type. Do not add a parallel hierarchy of preconditioner classes: the existing Pydantic config variants still select the algorithm, factory and schedule. The origin answers only the narrower question of what data built the method. If adding a generic origin object would merely mirror those variants without removing branching from presentation and cost, use the same closed cases in one resolver and keep its return type local to composition.

For checkpoint-backed POD-2G, compare the checkpoint run's recorded training-dataset digest with the digest of `dataset_dir` before any run opens. A recorded mismatch is a blocking provenance error. A run without a usable dataset identity, or an explicit checkpoint path without a run, is `TrainingDataUnavailable`: execute if the checkpoint itself is valid, omit the fit-dataset qualifier, and mark generation cost unavailable rather than charging the unverified directory. Do not read `dataset_dir` to reconstruct an already fitted POD basis. Inline POD continues to read snapshots and charge its verified dataset generation. Keep checkpoint load and data I/O outside timed setup as they are today.

Presentation is derived from the origin: `NoFitData` renders a standalone method; `InlineFit` and a single `VerifiedTrainingDataset` render a dataset-qualified method; unavailable provenance renders a standalone method plus an explicit provenance status in stored metadata and human-readable diagnostics. Do not append the comparison matrix set to any method label. At plot, MLflow run-name, logging and report-scope endpoints, render method and comparison scope separately through the existing functions. Preserve a distinct training/evaluation subject for assignment jobs, which genuinely has a training dataset.

## Steps (stop and report after each)

1. **Pin the current contract with tests.** In-memory tests cover `Identity`, Jacobi, IC0 and geometric AMG as no-fit-data; inline POD as fit-data; checkpoint POD with matching, mismatching and absent recorded digests; a neural checkpoint with verified and unavailable training provenance; neural POD's parameter dataset as runtime input, never training provenance. Assert exact method legend, method-run name and warning text separately from comparison scope text. Add one test that a group row still carries the comparison matrix-set id for Jacobi. Use fixtures in `conftest.py`, no real MLflow or image assertions.
2. **Resolve origin and provenance once.** Add the small closed type and resolver in `composition/comparison/`; use `current_dataset_digest` and existing run identity tags rather than inventing another hash. Feed it from `comparison_resolution.py` / `comparison_prepare.py` after checkpoint resolution and before presentation preflight, reuse checks or MLflow runs. Keep the source config as the only source of checkpoint paths. Update `ResolvedMethod` or the prepared execution value only as needed to carry the validated result. No default origin or partial intermediate state.
3. **Make cost attribution follow the validated origin.** Change `_offline_cost.py` to use the origin for generation applicability and the already resolved refs for training wall time. `NoFitData` is `NotApplicable`; inline fit reads its manifest; verified checkpoint training data reads only the attested dataset manifests; unavailable training data yields `Unavailable` generation cost. Keep `OfflineStages` and total-cost arithmetic unchanged. Test missing manifest duration and multi-checkpoint deduplication.
4. **Make presentation and every visible endpoint agree.** Change `presentation_preflight.py` to derive the existing presentation sum from the origin. Audit `run_names.py`, `figure_text.py`, `comparison_run.py`, report-scope labels, figure cards and training/evaluation legends for role wording and mixed-scope concatenation. Fix only paths that can imply a training relationship for no-fit-data methods. Assert that `Jacobi` and `Identity` appear alone as method labels while the comparison matrix set appears as an explicitly separate scope/title; verified POD fit data appears once; unknown provenance is visibly stated without a guessed dataset. Do not put ids or paths into scientific labels.
5. **Persist enough evidence to prevent false reuse.** Add the origin/provenance status and verified digest (when present) to the method's stored `presentation.toml` or one adjacent existing method artifact, with a strict reader and reuse-gate check against the freshly resolved origin. Derive it from the resolved value, never from a label. An old method run without the field, or one whose evidence differs from current resolution, must be rerun. Bump `REPORT_SCHEMA_VERSION` if figure cards, labels or report-run semantics change so an old report is not reused under new wording. Keep the existing comparison matrix-set event columns and report group schemas unchanged. Add a fake-store round trip and reuse rejection tests for both missing and changed evidence.
6. **Documentation and gates.** Update the affected module READMEs in `composition/comparison/`, `domain/presentation/`, `platform/reporting/` or `platform/tracking/` as code changes reach them, plus `composition/README.md` and `domain/solver/README.md` where cost and dataset roles are defined. Document the exact meanings of `fit dataset`, `training dataset`, `runtime parameter dataset`, and `comparison matrix set` next to their fields. Run `uv run pytest -m "integration or not integration"`, `uv run ruff check src tests`, `uv run ty check src/`, and `uv run tach check`. Then, on the compute machine, produce one new `compare` session and run `report`; hand-check a bare method, inline POD, checkpoint POD and neural method in the method run, log lines, plot legend and figure scope.

## Completion criteria and compatibility

- No-fit-data methods never carry a training-dataset presentation or an offline generation charge; they still have comparison matrix-set event rows.
- A fit/training dataset in a method label or cost is backed by validated provenance, and a recorded digest mismatch blocks before execution.
- Unknown checkpoint provenance is explicit in stored data and diagnostics; it is never inferred from a nearby path or the comparison matrix set.
- Runtime parameter data and comparison-system extra arrays are never described as the network's training data.
- Method, scope, training/evaluation and report labels use the same typed presentation rules at every endpoint.
- Existing checkpoint and system identities remain stable unless their actual inputs change. New method artifact provenance makes older method runs nonreusable; combine the required compute-machine rerun with plan 02 step 6 to avoid two experimental runs.

This plan does not change solver mathematics, dataset generation, TOML syntax, figure kinds or the meaning of matrix-set report groups. It does not add a universal `CheckpointPreconditioner | BarePreconditioner` wrapper: inline fitting and checkpoint-plus-runtime-data are real third and fourth cases, so that binary split would allow the confusion this plan removes.

## Implementation status (2026-11-10)

Steps 1–5 are implemented in `feat/method-provenance` and passed separate origin, cost, presentation, persistence, and whole-change review gates. Local validation passed: 3,780 tests, 7 skipped; `ruff check src tests`, `ty check src/`, `tach check`, and `git diff --check`. The full suite was run outside the restricted sandbox because zarr operations hang inside it; the isolated zarr case passed outside the sandbox.

Step 6's compute-machine `compare` and `report` run and visual hand-check remain pending. The new stored method evidence and report schema version intentionally reject reuse of older comparison/report artifacts.
