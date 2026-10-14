"""Up-front check that a preconditioner name means one preconditioner per dataset.

Setup sharing and scaling-law aggregation are both keyed on the preconditioner
name, so the same name used for different preconditioners would silently serve
or average unrelated builds. Nothing here reads a dataset or runs a comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from neuralls.composition.comparison.models import PlannedSystem
from neuralls.shared.digest import Digest

__all__ = ["validate_dataset_specs_consistent"]


def validate_dataset_specs_consistent(
    planned: Sequence[PlannedSystem],
    identities_by_entry_id: Mapping[str, Mapping[str, Digest]],
) -> None:
    """Reject a dataset whose systems use one preconditioner name for different preconditioners.

    "Different" is decided by identity (config digest plus checkpoint *content*
    digest, see `preconditioner_identities`), not by name or by `==`: a neural
    name only references a model, and the same model resolved to two temp paths
    is still one preconditioner. Systems absent from `identities_by_entry_id`
    (cache hits) are skipped — they run nothing this call.

    Raises:
        ValueError: Naming the dataset, the preconditioner, and both entries.
    """
    first_seen: dict[tuple[str, str], tuple[str, Digest]] = {}
    for system in planned:
        dataset = system.entry.matrix_dataset
        entry_id = system.entry.id
        for name, identity in identities_by_entry_id.get(entry_id, {}).items():
            previous = first_seen.setdefault((dataset, name), (entry_id, identity))
            if previous[1] != identity:
                raise ValueError(
                    f"Dataset '{dataset}': preconditioner '{name}' is configured "
                    f"differently in entries '{previous[0]}' and '{entry_id}' (different "
                    "settings or a different checkpoint behind the same name). Systems on "
                    "one dataset must share a preconditioner definition per name — rename it "
                    "or move one entry to a different dataset."
                )
