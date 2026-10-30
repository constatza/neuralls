"""Preconditioner configuration models.

These Pydantic models validate preconditioner configurations from TOML files.
They support both factory creation and scheduling/comparison concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

from neuralls.platform.config.models.checkpoint_refs import (
    NeuralCheckpointRef,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.models.coarsening import (
    CoarseningConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.fact_roles import IDENTITY_ROLE, OPERATIONAL_ROLE
from neuralls.platform.config.models.slugs import slug_number
from neuralls.shared.digest import Cosmetic
from neuralls.shared.fact_binding import (
    AFTER_METHOD_FACTS,
    Fact,
    Nested,
    Show,
    auto_if_none,
    enum_value,
    on_off,
)
from neuralls.shared.scientific_description.catalog import (
    ALGEBRAIC_DISTANCE_THRESHOLD,
    BOOTSTRAP_CYCLES,
    CALIBER,
    CALIBER_GROWTH_PENALTY,
    CANDIDATE_COUNT,
    CANDIDATE_ITERATIONS,
    COMPATIBLE_RELAXATION_SWEEPS,
    COMPATIBLE_RELAXATION_TOLERANCE,
    DAMPING,
    DROP_TOLERANCE,
    FALLBACK,
    ITERATION_LIMIT,
    MAX_COARSE_DIMENSION,
    MODEL_ARCHITECTURE,
    MULTIGRID_LEVELS,
    POST_SMOOTHING_STEPS,
    PRE_SMOOTHING_STEPS,
    RELAXATION_SWEEPS,
    RELAXATION_TEST_VECTORS,
    SEED,
    SMOOTHER_DAMPING,
    START_ITERATION,
    THRESHOLD,
    USE_LSR,
)
from neuralls.shared.scientific_description.models import ValidatedLabel


class MissingArchitectureNameError(ValueError):
    """A neural preconditioner has no ``architecture`` name."""

    def __init__(self) -> None:
        super().__init__(
            "A neural preconditioner needs a short 'architecture' name (for example 'Graph Net') "
            "so its architecture can be described; set it on the neural preconditioner or on "
            "the training job's [[jobs]] entry. Class names and job titles are never used."
        )


def architecture_label(name: str | None) -> ValidatedLabel:
    """The architecture name as a display label; a neural method cannot be described without one."""
    if name is None:
        raise MissingArchitectureNameError
    return ValidatedLabel(name)


class PreconditionerType(StrEnum):
    """Canonical preconditioner types accepted at the configuration boundary."""

    IDENTITY = "identity"
    JACOBI = "jacobi"
    ILU = "ilu"
    IC0 = "ic0"
    NEURAL = "neural"
    AMG = "amg"
    NEURAL_AMG = "neural_amg"
    ADAPTIVE_SA_AMG = "adaptive_sa_amg"
    BOOTSTRAP_AMG = "bootstrap_amg"

    def abbreviation(self) -> str:
        """Compact plot-legend base name for this type.

        Exhaustive ``match`` (no wildcard) so a future member forces a
        visible update here rather than silently falling through.
        ``AMG``/``NEURAL``/``NEURAL_AMG`` are unreachable via
        ``preconditioner_family()`` today (it collapses them to
        ``PreconditionerFamily`` members first), but are mapped here too for
        exhaustiveness/consistency if this enum is ever consumed directly.

        Returns:
            str: e.g. ``"aSA"``, ``"BAMG"``.
        """
        match self:
            case PreconditionerType.IDENTITY:
                return "Identity"
            case PreconditionerType.JACOBI:
                return "Jacobi"
            case PreconditionerType.ILU:
                return "ILU"
            case PreconditionerType.IC0:
                return "IC0"
            case PreconditionerType.NEURAL:
                return "Neural"
            case PreconditionerType.AMG:
                return "AMG"
            case PreconditionerType.NEURAL_AMG:
                return "Neural"
            case PreconditionerType.ADAPTIVE_SA_AMG:
                return "aSA"
            case PreconditionerType.BOOTSTRAP_AMG:
                return "BAMG"


class BasePreconditionerConfig(BaseModel, ABC):
    """Shared fields for all preconditioners.

    Includes scheduling fields shared by all preconditioner variants.
    """

    start_iter: Annotated[
        int, Fact(START_ITERATION, show=Show.NON_DEFAULT, order=AFTER_METHOD_FACTS)
    ] = Field(
        default=0,
        ge=0,
        description="Iteration at which the primary preconditioner becomes active.",
    )
    limit_iters: Annotated[
        int, Fact(ITERATION_LIMIT, show=Show.NON_DEFAULT, order=AFTER_METHOD_FACTS)
    ] = Field(default=-1, description="Iterations to apply; -1 means unlimited.")
    fallback: Annotated[
        PreconditionerType,
        Fact(FALLBACK, convert=enum_value, show=Show.NON_DEFAULT, order=AFTER_METHOD_FACTS),
    ] = Field(
        default=PreconditionerType.IDENTITY,
        description="Fallback preconditioner type when limited.",
    )

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields from conversion
        frozen=True,
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        """Deterministic result/plot key, derived from this config's own fields.

        Never user-settable: two configs with the same type and fields are
        the same preconditioner and must collide under this key (that
        collision is correct and is exactly what
        ``result_keys.validate_unique_preconditioner_keys`` catches); two
        configs differing in any distinguishing field always produce a
        different key. See ``slug`` — kept config-only
        (never reads a constructed object) so it's available before a
        preconditioner is built, for dict/cache keys and MLflow
        run-name/metric-key segments.
        """
        return self.slug()

    @abstractmethod
    def slug(self) -> str:
        """Deterministic identity key from this config's type and distinguishing fields."""


class StandardPreconditionerConfig(BasePreconditionerConfig):
    """Non-parametric, static preconditioners (identity, jacobi, ilu).

    Complexity: no setup beyond one factorization/diagonal extraction, no
    hyperparameters that change asymptotic cost. See the constructed
    ``torchalg`` class's own docstring for the exact storage/application
    Big-O (e.g. ``JacobiPreconditioner``: O(n) both; ``ILUPreconditioner``:
    O(nnz) both — derived from each class's actual tensor operations, not
    counted at runtime).
    """

    type: Annotated[
        Literal[
            PreconditionerType.IDENTITY,
            PreconditionerType.JACOBI,
            PreconditionerType.ILU,
        ],
        IDENTITY_ROLE,
    ]

    def slug(self) -> str:
        """Deterministic slug: the bare type name."""
        return self.type.value


class IC0PreconditionerConfig(BasePreconditionerConfig):
    """IC(0) preconditioner configuration with threshold parameter.

    Complexity: ``torchalg``'s ``IC0Preconditioner`` is a *dense masked* port
    (see its module docstring) — it stores and factorizes a dense ``n x n``
    buffer with the IC(0) sparsity mask applied, not a true sparse structure.
    Real cost is therefore O(n^2) storage and O(n^2) setup/application, not
    the sparse-textbook O(nnz) the IC(0) *algorithm* would suggest on a real
    sparse implementation — see ``torchalg.preconditioners.implementations
    .ic0.IC0Preconditioner`` for the derivation.
    """

    type: Annotated[Literal[PreconditionerType.IC0], IDENTITY_ROLE] = PreconditionerType.IC0
    threshold: Annotated[float, Fact(DROP_TOLERANCE)] = Field(
        default=0.0,
        description=(
            "Drop tolerance - entries with |value| <= threshold are treated as zeros. "
            "An absolute constant is unsafe here: comparisons run against a "
            "normalize_system-scaled matrix, so a fixed absolute threshold's "
            "effective strictness shifts with each matrix's own scale and can "
            "silently drop structurally-real entries, corrupting the IC(0) "
            "sparsity pattern. Default 0.0 only drops exact zeros."
        ),
    )

    def slug(self) -> str:
        """Deterministic slug for this config."""
        return f"ic0_threshold{slug_number(self.threshold)}"


class NeuralPreconditionerConfig(BasePreconditionerConfig, NeuralCheckpointRef):
    """Neural preconditioner configuration.

    Complexity: unlike every other preconditioner type in this file, cost is
    not derivable from this config alone — it is whatever the referenced
    checkpoint's network architecture costs per forward pass (setup: model
    load; application: one forward pass per CG iteration). No Big-O is stated
    here for that reason; measure it (``setup_cost``/``solve_time_seconds`` on
    the comparison result) rather than assume it.
    """

    type: Annotated[Literal[PreconditionerType.NEURAL], IDENTITY_ROLE] = PreconditionerType.NEURAL
    architecture: Annotated[
        str | None, Cosmetic(), Fact(MODEL_ARCHITECTURE, convert=architecture_label)
    ] = Field(
        default=None,
        description=(
            "Short human-readable architecture name shown in labels, e.g. 'Graph Net'. "
            "Required to describe the method; never read from a class name. Cosmetic, "
            "so it does not change the preconditioner's identity digest."
        ),
    )
    extra_input_names: Annotated[tuple[str, ...], OPERATIONAL_ROLE] = Field(
        default=(),
        description=(
            "Names of extra dataset arrays to bind before CG, beyond the residual. "
            "E.g. ('matrix',) for the stiffness matrix, ('coordinates',) for node coords. "
            "The comparison workflow loads matching named arrays from the dataset directory."
        ),
    )

    @field_validator("extra_input_names", mode="before")
    @classmethod
    def _coerce_extra_names_to_tuple(cls, v: object) -> tuple[str, ...] | object:
        """Coerce list to tuple for extra_input_names field.

        Args:
            v: Raw value that may be list, tuple, or other type.

        Returns:
            Tuple of strings, or original value if not list/tuple.
        """
        if isinstance(v, (list, tuple)):
            return tuple(str(x) for x in v)
        return v

    def checkpoint_refs(self) -> tuple[tuple[str, NeuralCheckpointRef], ...]:
        """This preconditioner's own identity is its single checkpoint ref."""
        return (("", self),)

    def with_resolved_refs(
        self, resolved: tuple[tuple[str, NeuralCheckpointRef], ...]
    ) -> NeuralPreconditionerConfig:
        """Resolution replaces this spec's own identity fields."""
        _, ref = resolved[0]
        return cast(NeuralPreconditionerConfig, ref)

    @model_validator(mode="after")
    def validate_single_model_identity_source(self) -> NeuralPreconditionerConfig:
        """Reject overlapping registered-model identity sources.

        `assignment` derives the registered model name automatically
        (`build_registered_model_name(assignment_id)`); `model_ref.name` on a
        `RegisteredModelRefConfig` names a registered model unrelated to any
        assignment. Setting both would leave undocumented, silent precedence
        between two competing identity sources.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If both `assignment` and a `RegisteredModelRefConfig`
                `model_ref.name` are set.
        """
        if (
            self.assignment is not None
            and isinstance(self.model_ref, RegisteredModelRefConfig)
            and self.model_ref.name is not None
        ):
            raise ValueError(
                "Set either 'assignment' (which derives the registered model name "
                "automatically) or 'model_ref.name' (for a registered model unrelated "
                "to any assignment) — not both; assignment is the single source of "
                "truth for an assignment's registered model identity."
            )
        return self

    def slug(self) -> str:
        """Deterministic slug for this config."""
        return f"neural_{self.checkpoint_slug()}"


def _has_algebraic_levels(cfg: AMGPreconditionerConfig) -> bool:
    """POD-2G is two-grid by construction, so its level count is not a parameter."""
    return not isinstance(cfg.coarsening, PODCoarseningConfig | NeuralPODCoarseningConfig)


class AMGPreconditionerConfig(BasePreconditionerConfig):
    """AMG-family preconditioner configuration (multigrid coarsening + cycle).

    ``coarsening`` selects the strategy that builds the prolongation/
    restriction operator — ``AggregationCoarseningConfig`` (classical SA-AMG),
    ``PODCoarseningConfig`` (POD-2G), or ``NeuralPODCoarseningConfig``
    (checkpoint-predicted POD-2G). Required, no default: the caller must
    state which coarsening strategy an AMG config means. A future coarsening
    strategy is added the same way: one more class in the
    ``CoarseningConfig`` union, not a new ``PreconditionerType`` and not new
    fields on this class — the underlying ``AMGPreconditioner`` is already
    generic over any ``CoarseningStrategy``.

    Complexity: application (one V-cycle) is **not** the textbook O(n) —
    that bound assumes sparse matvecs with a bounded coarsening ratio, but
    ``torchalg``'s AMG hierarchy is dense-only throughout: every level,
    transfer operator, and the coarsest-level solve via
    `torch.linalg.solve`/`pinv` all operate on dense `torch.Tensor` storage,
    never a sparse structure, regardless of the fine matrix's true sparsity.
    The finest level's residual matvec (`rhs - matrix @ x`,
    once per pre-smooth and post-smooth pass) is O(n^2), and dominates the
    cycle: coarser levels contribute smaller but still-dense O(level_size^2)
    terms, and the coarsest-level direct solve is O(coarsest_size^3) — see
    ``torchalg.preconditioners.implementations.amg.cycle``'s `VCycle._vcycle`
    for the actual recursion this is derived from. Setup cost is whichever
    coarsening strategy's own complexity note applies (``AggregationCoarseningConfig``
    / ``TargetDimCoarseningConfig`` / ``PODCoarseningConfig`` / ``NeuralPODCoarseningConfig``).
    """

    type: Annotated[Literal[PreconditionerType.AMG], IDENTITY_ROLE] = PreconditionerType.AMG
    n_levels: Annotated[int, Fact(MULTIGRID_LEVELS, when=_has_algebraic_levels)] = Field(
        default=2, ge=2, description="Total number of grid levels."
    )
    coarsening: Annotated[CoarseningConfig, Nested()]

    def checkpoint_refs(self) -> tuple[tuple[str, NeuralCheckpointRef], ...]:
        """Expose the coarsening's checkpoint ref, when it carries one.

        Each coarsening class answers for itself via ``exposed_checkpoint_ref``
        — a new coarsening strategy declares its own answer rather than adding
        a branch here.
        """
        ref = self.coarsening.exposed_checkpoint_ref()
        return (("coarsening", ref),) if ref is not None else ()

    def with_resolved_refs(
        self, resolved: tuple[tuple[str, NeuralCheckpointRef], ...]
    ) -> AMGPreconditionerConfig:
        """Rebuild with a resolved coarsening ref substituted in."""
        _, ref = resolved[0]
        return self.model_copy(update={"coarsening": cast(CoarseningConfig, ref)})

    pre_smoothing_steps: Annotated[int, Fact(PRE_SMOOTHING_STEPS, show=Show.NON_DEFAULT)] = Field(
        default=2, ge=0, description="Pre-smoothing iterations."
    )
    post_smoothing_steps: Annotated[int, Fact(POST_SMOOTHING_STEPS, show=Show.NON_DEFAULT)] = Field(
        default=2, ge=0, description="Post-smoothing iterations."
    )
    smoother_omega: Annotated[
        float | None, Fact(SMOOTHER_DAMPING, convert=auto_if_none, show=Show.NON_DEFAULT)
    ] = Field(
        default=None,
        gt=0.0,
        description="Weighted Jacobi damping; auto per-matrix via torchalg's "
        "spectral-radius rule when unset.",
    )

    def slug(self) -> str:
        """Deterministic slug for this config."""
        smoother_omega = (
            slug_number(self.smoother_omega) if self.smoother_omega is not None else "auto"
        )
        return (
            f"amg_levels{self.n_levels}_pre{self.pre_smoothing_steps}"
            f"_post{self.post_smoothing_steps}_smootheromega{smoother_omega}"
            f"_{self.coarsening.slug()}"
        )


class AdaptiveSAPreconditionerConfig(BasePreconditionerConfig):
    """Adaptive smoothed-aggregation AMG preconditioner (alpha-SA, PyAMG-faithful).

    Unlike ``AMGPreconditionerConfig``, this is not a pluggable-coarsening
    variant of the shared ``AMGPreconditioner(cycle=VCycle(JacobiSmoother))``
    path: ``torchalg.preconditioners.implementations.amg.AdaptiveSAPreconditioner``
    builds its whole multilevel hierarchy eagerly at construction (learned
    near-null-space candidates, symmetric Gauss-Seidel V(1,1) cycle, a
    pseudo-inverse coarse solve) and cannot be composed with
    ``AggregationCoarsening``/``TargetDimensionCoarsening`` the way POD-2G or
    classical SA-AMG can — hence its own ``PreconditionerType`` rather than
    another ``CoarseningConfig`` union member.

    ``theta`` should stay small for aSA — it is not the tuning knob for
    coarse-space size here, ``num_candidates`` is (see below). PyAMG's own
    default, independently confirmed against PyAMG's real source
    (``pyamg/aggregation/adaptive.py``'s ``adaptive_sa_solver`` ->
    ``pyamg/strength.py``'s ``symmetric_strength_of_connection(A, theta=0)``),
    is ``theta=0.0`` — "every non-zero coupling is strong", i.e. *maximal*
    aggregation/smallest coarse space, not "big": raising ``theta`` prunes
    the strength graph and makes the coarse space *larger*. There is no
    target-coarse-dimension search here (unlike classical SA-AMG's
    ``TargetDimCoarseningConfig``) — a real end-to-end run showed the
    search picking a *worse* theta than just naming one directly (its
    classical-AMG target-dim=10 search landed on a fully degenerate
    ``c=0``, while fixed ``theta=0.05`` alone gives ``c=23`` on the same
    504x504 matrix, see ``docs/plan.md``) — so pick ``theta`` directly
    instead, and keep it close to ``0`` (this repo's default is ``0.05``,
    the same value used at every tier; do not raise it to grow the coarse
    space — see ``num_candidates`` below).

    ``num_candidates`` is the actual coarse-space-size knob for aSA, not
    ``theta``: it defaults to ``3``, not PyAMG's own default of ``1``
    (PyAMG ships the cheapest, least-adaptive starting point and expects
    callers to raise it; its own examples typically use 2-4). Each
    additional candidate adds a DOF to every aggregate, so realized coarse
    dimension scales as ``(aggregate count at theta) x num_candidates`` —
    measured on a real 504x504 stiffness matrix from this repo
    (``SpectralData/45x15/stiffness/subdomain_1_Kaa.txt``) at a fixed
    ``theta=0.05``: ``c`` goes 69 (``k=3``) -> 115 (``k=5``) -> 161
    (``k=7``), so raising ``num_candidates`` alone reaches the same coarse
    dimension this repo's classical SA-AMG reaches by raising ``theta`` to
    ``0.25`` (also ``c=160``) — without ever leaving aSA's small-``theta``,
    maximal-aggregation regime, and while adding real multi-candidate
    adaptivity (more independent near-null-space directions) instead of
    just cruder aggregation.

    Complexity: setup repeats hierarchy construction once per candidate
    (`num_candidates` full aggregation + relaxation passes, `candidate_iters`
    relaxation sweeps each), so setup cost scales roughly linearly in
    `num_candidates` on top of one classical-SA-AMG-sized build — a real
    multiple, not free adaptivity. Application (V-cycle) is O(n^2)
    (dense-matrix-dominated, not the textbook sparse O(n)) once built —
    see `AMGPreconditionerConfig`'s complexity note.
    """

    type: Annotated[Literal[PreconditionerType.ADAPTIVE_SA_AMG], IDENTITY_ROLE] = (
        PreconditionerType.ADAPTIVE_SA_AMG
    )
    n_levels: Annotated[int, Fact(MULTIGRID_LEVELS)] = Field(
        default=2, ge=2, description="Cap on hierarchy depth (torchalg's `max_levels`)."
    )
    num_candidates: Annotated[int, Fact(CANDIDATE_COUNT)] = Field(
        default=3,
        ge=1,
        description="Total near-null-space candidates (including the initial one); "
        "PyAMG's own default is 1, but each extra candidate adds real adaptivity "
        "at the cost of one more DOF per aggregate in the realized coarse dimension.",
    )
    candidate_iters: Annotated[int, Fact(CANDIDATE_ITERATIONS, show=Show.NON_DEFAULT)] = Field(
        default=5, ge=1, description="Relaxation sweeps/cycles per candidate step."
    )
    max_coarse: Annotated[int, Fact(MAX_COARSE_DIMENSION, show=Show.NON_DEFAULT)] = Field(
        default=10, ge=1, description="Stop coarsening once a level has this many nodes or fewer."
    )
    theta: Annotated[float, Fact(THRESHOLD)] = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Strength-of-connection threshold — pick directly, no search; "
        "close to 0 for a small coarse space, ~0.25 for a larger one.",
    )
    omega: Annotated[float, Fact(DAMPING, show=Show.NON_DEFAULT)] = Field(
        default=4.0 / 3.0, gt=0.0, description="Nominal Jacobi prolongator-smoothing damping."
    )
    seed: Annotated[int, Fact(SEED)] = Field(
        default=0, description="Seed for the candidate/spectral-radius random source."
    )

    def slug(self) -> str:
        """Deterministic slug for this config."""
        return (
            f"asa_levels{self.n_levels}_k{self.num_candidates}_iters{self.candidate_iters}"
            f"_maxcoarse{self.max_coarse}_theta{slug_number(self.theta)}"
            f"_omega{slug_number(self.omega)}_seed{self.seed}"
        )


class BootstrapAMGPreconditionerConfig(BasePreconditionerConfig):
    """Bootstrap AMG preconditioner (BAMG), ``torchalg``'s ``BootstrapAMGPreconditioner``.

    Like ``AdaptiveSAPreconditionerConfig``, this builds its whole multilevel
    hierarchy eagerly at construction (compatible-relaxation coarsening,
    algebraic-distance interpolation, a bootstrap-cycle-fitted test-vector
    set) and is not a pluggable ``CoarseningStrategy`` the way classical
    SA-AMG or POD-2G are — hence its own ``PreconditionerType`` rather than
    another ``CoarseningConfig`` union member. Field defaults mirror
    ``torchalg.preconditioners.implementations.amg.bootstrap
    .BootstrapAMGPreconditioner``'s own constructor defaults, including
    ``n_levels`` (torchalg's ``max_levels=10``).

    Unlike ``AMGPreconditionerConfig``'s ``n_levels=2``, this is not
    truncated to two-grid: that default exists to match POD-2G, which is
    architecturally single-basis two-grid and can't be anything else
    (``ADAPTIVE_SA_AMG``/``NEURAL_AMG`` inherited the same default, but
    nothing ties it to a general rule that every AMG-family method must
    default to two-grid). ``n_levels`` here is only a *ceiling* on
    ``BootstrapSetup``'s coarsening loop
    (``while len(levels) < max_levels and levels[-1].shape[0] >
    max_coarse``) — the loop actually stops once a level's dimension drops
    to ``max_coarse`` or below, whichever comes first, so realized depth on
    this repo's matrices (a few hundred DOF) is typically well under 10 and
    the cap rarely binds. Truncating it to 2 would defeat the point of
    BAMG's setup cost (several multiples of a classical-AMG setup, see the
    torchalg docstring) for no deeper a hierarchy than cheap classical AMG
    already gives.

    Complexity: setup is `n_bootstrap_cycles` full bootstrap passes, each
    building a test-vector set (`k_r` vectors x `eta` relaxation sweeps) plus
    a compatible-relaxation coarsening and algebraic-distance interpolation
    per level — several multiples of classical AMG's single-pass setup, as
    the class docstring above already states. Application (V-cycle) is
    O(n^2) (dense-matrix-dominated, not the textbook sparse O(n)) once
    built — see `AMGPreconditionerConfig`'s complexity note.
    """

    type: Annotated[Literal[PreconditionerType.BOOTSTRAP_AMG], IDENTITY_ROLE] = (
        PreconditionerType.BOOTSTRAP_AMG
    )
    n_levels: Annotated[int, Fact(MULTIGRID_LEVELS)] = Field(
        default=10,
        ge=2,
        description="Ceiling on hierarchy depth (torchalg's `max_levels`); the coarsening "
        "loop actually stops at `max_coarse`, whichever comes first, so this rarely binds.",
    )
    k_r: Annotated[int, Fact(RELAXATION_TEST_VECTORS, show=Show.NON_DEFAULT)] = Field(
        default=8, ge=1, description="Number of relaxation-derived test vectors."
    )
    eta: Annotated[int, Fact(RELAXATION_SWEEPS, show=Show.NON_DEFAULT)] = Field(
        default=4, ge=1, description="Relaxation sweeps per test vector per level."
    )
    n_bootstrap_cycles: Annotated[int, Fact(BOOTSTRAP_CYCLES, show=Show.NON_DEFAULT)] = Field(
        default=2, ge=1, description="Number of bootstrap-cycle passes."
    )
    nu: Annotated[int, Fact(COMPATIBLE_RELAXATION_SWEEPS, show=Show.NON_DEFAULT)] = Field(
        default=5, ge=1, description="CR sweeps per stage."
    )
    delta: Annotated[float, Fact(COMPATIBLE_RELAXATION_TOLERANCE, show=Show.NON_DEFAULT)] = Field(
        default=0.7, gt=0.0, description="CR stopping tolerance."
    )
    theta_ad: Annotated[float, Fact(ALGEBRAIC_DISTANCE_THRESHOLD, show=Show.NON_DEFAULT)] = Field(
        default=0.5, ge=0.0, description="Algebraic-distance strength threshold."
    )
    caliber: Annotated[int, Fact(CALIBER, show=Show.NON_DEFAULT)] = Field(
        default=4, ge=1, description="Maximum interpolatory-set size."
    )
    gamma: Annotated[float, Fact(CALIBER_GROWTH_PENALTY, show=Show.NON_DEFAULT)] = Field(
        default=1.5, gt=0.0, description="Caliber-growth penalization exponent."
    )
    use_lsr: Annotated[bool, Fact(USE_LSR, convert=on_off, show=Show.NON_DEFAULT)] = Field(
        default=True, description="Apply the LSR residual correction before fitting."
    )
    max_coarse: Annotated[int, Fact(MAX_COARSE_DIMENSION, show=Show.NON_DEFAULT)] = Field(
        default=10, ge=1, description="Stop coarsening at this many nodes."
    )
    seed: Annotated[int, Fact(SEED)] = Field(
        default=0, description="Seed of the random test-vector source."
    )

    def slug(self) -> str:
        """Deterministic slug for this config."""
        return (
            f"bamg_levels{self.n_levels}_kr{self.k_r}_eta{self.eta}"
            f"_cycles{self.n_bootstrap_cycles}_nu{self.nu}"
            f"_delta{slug_number(self.delta)}_thetaad{slug_number(self.theta_ad)}"
            f"_caliber{self.caliber}_gamma{slug_number(self.gamma)}_lsr{self.use_lsr}"
            f"_maxcoarse{self.max_coarse}_seed{self.seed}"
        )


class NeuralTransferConfig(NeuralCheckpointRef):
    """Config for one neural transfer operator (prolongation or restriction).

    Required inputs (the extra arrays the network expects beyond its vector input)
    are NOT declared here — they are read at runtime from
    ``ExtraInputPredictorPort.required_inputs``, which is populated by the adapter
    from the model config.  This eliminates duplication with the DLKit TOML.
    """


class NeuralAMGPreconditionerConfig(BasePreconditionerConfig):
    """AMG preconditioner that uses neural networks for prolongation/restriction.

    The ``extra_input_names`` required by each network are NOT declared here —
    they are read at runtime from ``ExtraInputPredictorPort.required_inputs`` so
    that the DLKit model config remains the single source of truth.

    Complexity: like ``NeuralPreconditionerConfig``, transfer-operator cost
    (setup: model load; application: one forward pass per level per V-cycle)
    is architecture-dependent and not stated here. The surrounding cycle's
    own linear-algebra work is O(n^2) per V-cycle (dense-matrix-dominated —
    see ``AMGPreconditionerConfig``'s complexity note), on top of whatever
    the neural transfer operators' own forward passes cost.
    """

    type: Annotated[Literal[PreconditionerType.NEURAL_AMG], IDENTITY_ROLE] = (
        PreconditionerType.NEURAL_AMG
    )
    n_levels: Annotated[int, Fact(MULTIGRID_LEVELS)] = Field(
        default=2, ge=2, description="Total number of grid levels."
    )
    aggregation_omega: Annotated[float | None, Fact(DAMPING, show=Show.NOT_NONE)] = Field(
        default=None, gt=0.0
    )
    pre_smoothing_steps: Annotated[int, Fact(PRE_SMOOTHING_STEPS, show=Show.NON_DEFAULT)] = Field(
        default=2, ge=0
    )
    post_smoothing_steps: Annotated[int, Fact(POST_SMOOTHING_STEPS, show=Show.NON_DEFAULT)] = Field(
        default=2, ge=0
    )
    smoother_omega: Annotated[
        float | None, Fact(SMOOTHER_DAMPING, convert=auto_if_none, show=Show.NON_DEFAULT)
    ] = Field(default=None, gt=0.0)
    prolongation: Annotated[NeuralTransferConfig, IDENTITY_ROLE]
    restriction: Annotated[NeuralTransferConfig | None, IDENTITY_ROLE] = None

    def checkpoint_refs(self) -> tuple[tuple[str, NeuralCheckpointRef], ...]:
        """Expose prolongation's ref, plus restriction's when set."""
        refs: tuple[tuple[str, NeuralCheckpointRef], ...] = (("prolongation", self.prolongation),)
        if self.restriction is not None:
            refs += (("restriction", self.restriction),)
        return refs

    def with_resolved_refs(
        self, resolved: tuple[tuple[str, NeuralCheckpointRef], ...]
    ) -> NeuralAMGPreconditionerConfig:
        """Rebuild with resolved prolongation/restriction refs substituted in."""
        updates: dict[str, NeuralTransferConfig] = {}
        for label, ref in resolved:
            updates[label] = cast(NeuralTransferConfig, ref)
        return self.model_copy(update=updates)

    def slug(self) -> str:
        """Deterministic slug for this config."""
        smoother_omega = (
            slug_number(self.smoother_omega) if self.smoother_omega is not None else "auto"
        )
        aggregation_omega = (
            slug_number(self.aggregation_omega) if self.aggregation_omega is not None else "auto"
        )
        base = (
            f"neural_amg_levels{self.n_levels}_pre{self.pre_smoothing_steps}"
            f"_post{self.post_smoothing_steps}_smootheromega{smoother_omega}"
            f"_aggomega{aggregation_omega}_{self.prolongation.checkpoint_slug()}"
        )
        if self.restriction is None:
            return base
        return f"{base}_{self.restriction.checkpoint_slug()}"


ConcretePreconditionerConfig = (
    StandardPreconditionerConfig
    | IC0PreconditionerConfig
    | NeuralPreconditionerConfig
    | AMGPreconditionerConfig
    | NeuralAMGPreconditionerConfig
    | AdaptiveSAPreconditionerConfig
    | BootstrapAMGPreconditionerConfig
)

_StrictPreconditionerConfig = Annotated[
    ConcretePreconditionerConfig,
    Field(discriminator="type"),
]

PreconditionerConfig = _StrictPreconditionerConfig

parse_preconditioner_config = TypeAdapter(PreconditionerConfig).validate_python
