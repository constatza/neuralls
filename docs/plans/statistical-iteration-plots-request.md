# Request: statistical iteration plots comparable to arXiv:2605.19867v3

Please provide comparable statistical plots for `spectral-3d-field` using
Melchers, Dolean and Abdelmalik, *When can a neural operator replace a coarse
solve? Architectural principles for two-level preconditioning*,
[arXiv:2605.19867v3](https://arxiv.org/abs/2605.19867v3), as a reporting precedent.
This is an output request, not an implementation plan. The developers decide
how to aggregate, render and integrate the outputs with existing reporting.

## What their per-dataset statistical plots do

- **Figure 2 (PDF page 12):** one panel per diffusion training-source dataset,
  plus a fourth boundary-basis variant. Within each panel, mesh resolutions
  form the horizontal groups, neighboring bars compare methods, and bar
  heights are mean F-GMRES iterations over 1000 held-out realizations. A common
  method legend, logarithmic iteration scale and dashed iteration limit allow
  comparisons across panels. Consistently failing entries are omitted and
  described explicitly.
- **Important distinction:** short colored horizontal marks above the bars
  are another experiment, not min–max or quartile error bars. The bars use
  in-distribution RHS vectors and a mesh-dependent tolerance; the short marks
  use independent Gaussian algebraic RHS vectors and tolerance `1e-10`.
- **Figure 5 (PDF page 14):** separately fixes Dataset 3 and the 80-by-80 mesh
  and compares each method's iteration distribution over 1000 test systems.
  Horizontal boxplots show the median, 25th–75th percentile box and
  5th–95th percentile whiskers, with observations above the 95th percentile
  plotted individually. These are distribution summaries, not confidence
  intervals or min–max whiskers. Figure 14 uses boxplots to compare training
  sample counts, though its caption does not repeat the whisker convention.
- **Failures matter:** the paper warns that means hide tails and failures.
  Tables 4/5 explicitly give means conditional on convergence alongside
  failure counts; do not assume every earlier source figure uses that same
  averaging rule without checking its experiment description.

## Their RHS/dataset strategy and its relation to our case

The three diffusion training-source distributions are smooth Gaussian random
fields, Gaussian-coefficient expansions in the common B-spline basis, and
inverse-Gram-transformed Gaussian coefficient draws in that basis. The
diffusivity distribution is held the same. Each architecture is trained and
tested on its own source distribution. These are therefore construction-data
panels as well as evaluation distributions, not four target RHS recipes
applied to one unchanged learned model. The independent Gaussian-RHS overlay
is a second robustness test and also changes tolerance.

Our current case has one evaluation matrix set, five held-out matrices, four
target RHS recipes per matrix and eleven candidate method configurations.
The two snapshot ensembles identify constructed methods; the four target RHS
recipes identify evaluation cohorts. Please preserve that distinction and
the existing common tolerance when producing similar figures. We have no
mesh-resolution sweep to reproduce their mesh groups literally.

Desired outputs are per-cohort/per-dataset mean-iteration comparisons and
distribution views that expose variation between target systems, with clear
sample sizes, method identity and failure treatment. Retain the existing
all-planned ECDFs and success counts as complementary reliability evidence.
Five observations per RHS cohort call for showing the actual observations
and qualifying percentile summaries; timing repeats are not extra independent
systems. Entirely unsuccessful methods and unavailable statistics must remain
identifiable. Exact layouts and implementation choices are left to the team.

## Comparison levels and prerequisites for our cases

These are scientific comparison dimensions and desired outputs, not a required
run hierarchy or implementation plan. Please decide how they fit the existing
reporting system.

| Comparison dimension | What we already have in spectral 3D | What further data are needed / what the plot means |
|---|---|---|
| Target RHS distributions | Four recipes on the same five matrices | No new dataset needed. Compare methods separately by RHS class using mean/distribution plots, ECDFs and histories at the same tolerance. More independent realizations improve evidence; timing repeats are not extra systems. |
| Snapshot-construction datasets | Two ensembles defining four POD method configurations | No new evaluation set needed. These identify method variants evaluated on the same target systems, not two evaluation datasets. Keep construction provenance and costs visible. |
| Evaluation datasets / physical problem families | One evaluation matrix set | At least one additional genuinely distinct evaluated matrix set is needed. It can be another set within a case or a compatible separate case. Prefer dataset-specific panels with comparable methods and identifiable RHS classes. More samples from the same distribution increase sample size, not dataset count. |
| Physical parameters / regimes | Several material realizations; parameter metadata is not yet verified | Record/recover each realization's physical parameters and obtain sufficient range coverage. This enables iterations/cost versus parameter scatter or predefined regime summaries. Do not treat five unidentified matrices as a parameter-robustness study. |
| Mesh resolution / matrix size | No resolution sweep | Actual multiple resolutions of a documented problem and compatible solve conditions are needed. Group methods within resolutions. POD generally needs a separate basis for each resolution because discrete spaces change. |
| Overall suite across datasets/cases | Current global and dataset populations are the same | Multiple compatible evaluation sets, explicit dataset/RHS weights and defined cost/accuracy scopes. Preserve per-set views; avoid silently weighting by sample count or averaging incomparable raw runtimes. |

Each added evaluation set should identify its physical family, matrix ids,
parameter/resolution information, RHS recipes, construction/test separation,
methods, outcome counts and measurement scope. Accuracy criteria must be
comparable and normalization/precision/timing conditions stated. Method
settings may legitimately differ by case, but that must be labelled. Pair
methods on the same targets within each set and qualify matrix identity by
its dataset. Do not assume separate compute sessions already form a compatible
cross-case analysis.

Keep solve-only, online and full total/amortized costs distinct. Shared
snapshot/POD cost is charged once per deployed basis; a new set or resolution
requiring another basis incurs its own construction cost. Matrix-specific
setup is still charged for each matrix. A suite-level cost plot needs a clear
deployment/reuse scenario and identifiable failed/unattempted systems.

For the present case, the directly applicable source families are mean bars,
boxplot/distribution views, representative convergence histories and timing
comparisons/component breakdowns; existing ECDFs and success counts remain
complementary. Mesh/material illustrations need physical exports. Parameter
scatter and cross-dataset/resolution figures are conditional extensions with
the prerequisites above. Spectral/operator diagnostics are optional separate
work, not a requirement; neural architecture/training studies and standalone
POD-spectrum validation remain outside the current experiment scope.

## Complete inventory of plot/display types in the source

| Type | Source figures | What they encode |
|---|---|---|
| Source-function sample curves | 1 | Representative draws from three source distributions. |
| Grouped mean-iteration bars, sometimes combined with other displays | 2, 8, 9, 10, 15 | Dataset/mesh/method comparisons, architecture/combination-form comparison, or out-of-distribution comparison. Figures 2/10 also overlay the separate RHS/tolerance experiment. |
| Spatial preconditioner-input images | 3 | Evolution of the vectors presented to a preconditioner during Krylov iteration. |
| Representative relative-residual histories | 4, 8, 12, 17 | Qualitative convergence/stagnation of methods on selected systems and PDE regimes. |
| Iteration-distribution boxplots | 5, 14 | Variation across test systems, by method or training sample count. |
| Complex-plane eigenvalue scatter grids | 6, 13, 16 | Preconditioned spectra across methods/PDE regimes, with commutator-error annotations. |
| Green-function/operator-kernel heatmaps | 7 | Exact versus learned correction structure. |
| Parameter-versus-iteration scatter | 11, 15 | Per-realization iteration behavior versus Péclet number or wave number. |
| Average timing comparisons and stacked component costs | 18 | Timing across mesh/method and fixed-mesh construction/application breakdown; Table 6 also gives iterations and timing components. |
| PDE coefficient/source/solution field images and boundary-data curves | 9, 19, 20, 21 | Example inputs/solutions, in-/out-of-distribution regimes, velocity overlays and complex-valued field components where relevant. |
| Smoother-kernel surface | 22 | Three-dimensional visualization of the convolution smoother kernel. |
| Smoother sweep-direction diagrams | 23 | The four spatial sweep directions. |
| Neural-architecture/dataflow diagrams | 24 | Input processing, learned operators and output combination. |

All figure numbers 1–24 are covered above. Numerical Tables 2–5 complement
the figures with means and failures; Table 6 gives timing/iteration summaries;
Tables 1 and 7 describe architecture classes and training settings. The paper
does not show ECDFs or Dolan–Moré performance profiles.

This catalog records all source display types; it is not a request to add every
type to our current case. The priority is comparable mean/distribution plots
for solver performance. Neural diagnostics and POD-spectrum validation are
outside our paper's current experiment scope. No implementation steps, module
changes, new schema or new experiment are prescribed here.
