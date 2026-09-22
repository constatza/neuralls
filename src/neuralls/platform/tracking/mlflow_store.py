"""MLflow-backed `IdentityStore`: the single reader of reusable runs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from dlkit.mlflow import has_checkpoint_artifact
from loguru import logger
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from neuralls.domain.identity import IdentityTag, Reused, StageIdentity
from neuralls.platform.tracking.mlflow import quote_filter_value

_PAGE_SIZE = 50
_NEWEST_FIRST = ["attributes.start_time DESC", "attributes.run_id ASC"]


def _tag_clause(tag: IdentityTag, value: str) -> str:
    return f"tags.`{tag}` = '{quote_filter_value(value)}'"


@dataclass(frozen=True)
class MlflowIdentityStore:
    """Finds the newest FINISHED run in one experiment carrying an identity's key.

    Attributes:
        tracking_uri (str): MLflow tracking URI.
        experiment (str): Experiment name the stage's runs live in; runs from
            any other experiment are never considered.
        require_checkpoint (bool): Also require a real checkpoint artifact
            (training runs); a FINISHED run without one is skipped in favour
            of an older run with the same key.
    """

    tracking_uri: str
    experiment: str
    require_checkpoint: bool = False

    def find(self, identity: StageIdentity) -> Reused | None:
        """Return the newest matching run, or ``None``.

        Ordering is deterministic (start time, then run id) and pagination
        walks every match, so a pile of newer broken runs cannot hide an older
        good one.
        """
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment)
        if experiment is None:
            return None
        for run in self._matching_runs(client, experiment.experiment_id, identity):
            run_id = run.info.run_id
            if not self.require_checkpoint or has_checkpoint_artifact(
                run_id, tracking_uri=self.tracking_uri
            ):
                return Reused(run_id=run_id)
            logger.warning(
                "Run {} matches {} identity {} but has no checkpoint artifact; trying an older run.",
                run_id,
                identity.stage,
                identity.key[:19],
            )
        return None

    def _matching_runs(
        self, client: MlflowClient, experiment_id: str, identity: StageIdentity
    ) -> Iterator[Run]:
        filter_string = " and ".join(
            [
                "attributes.status = 'FINISHED'",
                _tag_clause(IdentityTag.STAGE, identity.stage),
                _tag_clause(IdentityTag.KEY, identity.key),
            ]
        )
        token: str | None = None
        while True:
            page = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                order_by=_NEWEST_FIRST,
                max_results=_PAGE_SIZE,
                page_token=token,
            )
            yield from page
            token = page.token
            if not token:
                return
