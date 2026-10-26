"""Pure helpers for auto-id generation in case configs.

This module provides functions to infer missing 'id' fields
in assignment and comparison entries, with sensible fallback logic and validation.
No Pydantic imports — only plain Python and regex.
"""

from __future__ import annotations

import re

_ID_PATTERN = re.compile(r"^[\w\-]+$")


def _validate_id_chars(id_val: str, context: str = "id") -> None:
    r"""Raise ValueError if id_val contains characters outside ^[\w\-]+$.

    Args:
        id_val: The ID string to validate.
        context: Description of what is being validated (for error messages).

    Raises:
        ValueError: If id_val contains invalid characters.
    """
    if not _ID_PATTERN.fullmatch(id_val):
        raise ValueError(
            f"{context} contains invalid characters: {id_val!r} must match ^[\\w\\-]+$"
        )


def _slugify(display_name: str) -> str:
    """Lowercase, replace spaces with hyphens, validate result against _ID_PATTERN.

    Args:
        display_name: The display name to slugify.

    Returns:
        The slugified ID string.

    Raises:
        ValueError: If the result is invalid after slugification, or if input is empty.
    """
    if not display_name:
        raise ValueError("invalid display_name: cannot be empty")

    # Strip surrounding whitespace, lowercase, and replace spaces with hyphens
    slugified = display_name.strip().lower().replace(" ", "-")

    # Validate the result
    if not _ID_PATTERN.fullmatch(slugified):
        raise ValueError(f"invalid id after slugification: {slugified!r}")

    return slugified


def _infer_assignment_id(
    dataset_id: str,
    job_id: str,
    user_id: str | None,
    user_dn: str | None,
) -> str:
    """Return the resolved assignment id.

    Priority:
      1. user_id (stripped, if non-blank)
      2. _slugify(user_dn) if user_dn non-blank
      3. f"{job_id}-{dataset_id}"  (job first)

    Validates result with _validate_id_chars before returning.

    Args:
        dataset_id: The dataset identifier.
        job_id: The job identifier.
        user_id: Explicit user-supplied id (may be None or blank).
        user_dn: Explicit user-supplied display_name (may be None or blank).

    Returns:
        The resolved assignment id string.

    Raises:
        ValueError: If the resolved id contains invalid characters.
    """
    if user_id and user_id.strip():
        final_id = user_id.strip()
    elif user_dn and user_dn.strip():
        final_id = _slugify(user_dn)
    else:
        final_id = f"{job_id}-{dataset_id}"

    _validate_id_chars(final_id, "id")
    return final_id


def _infer_comparison_id(
    matrix_id: str,
    rhs_id: str,
    user_id: str | None,
    user_dn: str | None,
) -> str:
    """Return the resolved comparison id.

    Priority:
      1. user_id (stripped, if non-blank)
      2. _slugify(user_dn) if user_dn non-blank
      3. matrix_id           (when matrix_id == rhs_id)
         f"{matrix_id}-{rhs_id}"  otherwise (single dash)

    Validates result with _validate_id_chars before returning.

    Args:
        matrix_id: The matrix dataset identifier.
        rhs_id: The right-hand side dataset identifier.
        user_id: Explicit user-supplied id (may be None or blank).
        user_dn: Explicit user-supplied display_name (may be None or blank).

    Returns:
        The resolved comparison id string.

    Raises:
        ValueError: If the resolved id contains invalid characters.
    """
    if user_id and user_id.strip():
        final_id = user_id.strip()
    elif user_dn and user_dn.strip():
        final_id = _slugify(user_dn)
    elif matrix_id == rhs_id:
        final_id = matrix_id
    else:
        final_id = f"{matrix_id}-{rhs_id}"

    _validate_id_chars(final_id, "id")
    return final_id
