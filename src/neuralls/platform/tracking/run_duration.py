"""Wall time of a finished MLflow run, as a stage cost."""

from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from neuralls.domain.solver.models.cost import Known, Unavailable
from neuralls.shared.types import CostProvenance

_MS_PER_S = 1000.0


def run_wall_seconds(client: MlflowClient, run_id: str) -> Known | Unavailable:
    """The seconds from the start to the end of run ``run_id``, charged back as `HISTORICAL`.

    The span covers everything the run did while open (data loading, epochs, checkpointing),
    not only the fit. A run the store does not hold or one that has not ended has no
    duration, so it is `Unavailable` and never a zero.
    """
    try:
        info = client.get_run(run_id).info
    except MlflowException as error:
        return Unavailable(f"run {run_id} cannot be read: {error.message}")
    if info.end_time is None:
        return Unavailable(f"run {run_id} has no end time")
    return Known((info.end_time - info.start_time) / _MS_PER_S, None, CostProvenance.HISTORICAL)
