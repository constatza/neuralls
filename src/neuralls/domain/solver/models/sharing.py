"""Which setup and generation costs amortize across sibling systems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SetupSharingPolicy:
    """Whether one preconditioner's setup/generation cost amortizes across sibling systems.

    Two independent facts, not a single state — a preconditioner can share
    one without the other (e.g. a checkpointless POD-2G AMG preconditioner
    in a fixed-matrix sweep: its `AMGPreconditioner.setup(matrix)` is
    shareable because the matrix never changes, but its inline-fit
    generation cost reruns every comparison regardless).

    Attributes:
        share_setup: True iff the preconditioner's `setup()` is provably
            reusable across the sibling systems being aggregated — either
            because it is a `torchalg.StaticSetupPreconditioner` (setup
            never reads the matrix at all) or because every sibling resolves
            to the same matrix (an RHS-only sweep).
        share_generation: True iff `GenerationRepeatability.ONE_TIME` for
            this preconditioner's config — its fit/train step already
            happened once, upstream, and is not being repeated here.
    """

    share_setup: bool
    share_generation: bool
