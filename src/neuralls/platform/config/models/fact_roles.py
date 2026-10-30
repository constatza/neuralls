"""Fact roles shared by the preconditioner config modules.

Each marker tells the scientific-description renderer that a field is
recorded under a role (identity, provenance, operational) instead of being
rendered as a method fact.
"""

from __future__ import annotations

from neuralls.shared.fact_binding import Omitted
from neuralls.shared.scientific_description.catalog import FactRole

IDENTITY_ROLE = Omitted(FactRole.IDENTITY)
PROVENANCE_ROLE = Omitted(FactRole.PROVENANCE)
OPERATIONAL_ROLE = Omitted(FactRole.OPERATIONAL)
