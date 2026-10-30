"""In-memory ``RunTree`` recording the child runs a report would create."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CreatedRun:
    run_id: str
    parent_run_id: str
    name: str
    tags: Mapping[str, str]
    params: Mapping[str, str]


@dataclass(slots=True)
class FakeRunTree:
    created: list[CreatedRun] = field(default_factory=list)

    def child_run(
        self, parent_run_id: str, name: str, tags: Mapping[str, str], params: Mapping[str, str]
    ) -> str:
        run_id = f"child-{len(self.created) + 1}"
        self.created.append(CreatedRun(run_id, parent_run_id, name, tags, params))
        return run_id
