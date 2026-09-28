"""Temporary exceptions to the operand-hierarchy invariant.

Temporary exceptions; the target is an empty set. Each key permits a higher-tier forward
operand the checker otherwise rejects, and each is deleted once the overload it permits stops
naming that operand. The checker fails, in the ``architecture`` CI job, on a key no scanned
stub site matched, so a key cannot outlive its overload.

The list lives apart from the checker so that it reads as the temporary list it is. It carries
no per-key rationale and no per-key documentation: the reason is argued in the review and the
commit that add a key, the policy lives in
``docs/type-architecture/operand-hierarchy.md``, and ``TEMPORARY_EXCEPTION_NOTE`` is the one
wording the checker prints.
"""

from __future__ import annotations

from typing import Final

ExceptionKey = tuple[str, str, str]

# The one shared wording for this list: the module docstring's first sentence and the
# sentence the checker appends to its pass summary. Do not add per-key prose.
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
