"""Temporary exceptions to the operand-hierarchy invariant.

Temporary exceptions; the target is an empty set. Each key permits a higher-tier forward
operand the checker otherwise rejects, and is deleted once the overload it permits stops
naming that operand. The checker fails, in the ``architecture`` CI job, on a key no scanned
stub site matched, so a key cannot outlive its overload.

This module is the list alone: no per-key rationale and no per-key anchor. The policy lives
in ``docs/type-architecture/operand-hierarchy.md``, the reason for a key is argued in the
review and the commit that adds it, and ``TEMPORARY_EXCEPTION_NOTE`` is the one wording the
checker prints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ExceptionKey = tuple[str, str, str]

# The path this file is reached by, derived rather than written down twice, so the debt line
# and the dead-key message cannot outlive a move of this file. The name is computed from the
# module's own location: ``scripts/operand_hierarchy/exceptions.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
EXCEPTIONS_FILE: Final[str] = (
    Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix()
)

# The one shared wording for this list: quoted by the module docstring above and appended
# by the checker to its pass summary. Do not add per-key prose.
TEMPORARY_EXCEPTION_NOTE: Final[str] = (
    "Temporary exceptions; the target is an empty set"
)

# Where the policy and the target are documented. One anchor, not one per key.
EXCEPTIONS_DOCUMENTATION: Final[str] = (
    "docs/type-architecture/operand-hierarchy.md#scalar-and-index-subclass-overloads"
)

# A key of (class, "*", forbidden) permits `forbidden` for every forward binary dunder of
# `class`; an exact (class, dunder, forbidden) key is looked up first and wins. `forbidden` is
# the tier's operand name, so the Index key covers the IntervalIndex, TimedeltaIndex and
# MultiIndex spellings alike.
FORWARD_DUNDER_EXCEPTIONS: Final[frozenset[ExceptionKey]] = frozenset(
    {
        ("Interval", "*", "Index"),
        ("Interval", "*", "Series"),
        ("IntervalIndex", "*", "Series"),
        ("NAType", "*", "Index"),
        ("NAType", "*", "Series"),
        ("Period", "*", "Index"),
        ("Period", "*", "Series"),
        ("Series", "__matmul__", "DataFrame"),
        ("Timedelta", "*", "Index"),
        ("Timedelta", "*", "Series"),
        ("Timestamp", "*", "Index"),
        ("Timestamp", "*", "Series"),
    }
)
