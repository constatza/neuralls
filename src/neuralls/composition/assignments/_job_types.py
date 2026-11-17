"""Shared job types for assignment composition."""

from __future__ import annotations

from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

from neuralls.platform.config.models.pod_job import PODJobConfig

# Jobs dlkit runs: every runtime patch, dataset and tracking stage works on these.
type DlkitJobConfig = TrainingJobConfig | InferenceJobConfig | SearchJobConfig
# Any job an assignment can declare; a POD recipe is fitted inside the comparison.
type AnyJobConfig = DlkitJobConfig | PODJobConfig
type TrainLikeJobConfig = TrainingJobConfig | SearchJobConfig
