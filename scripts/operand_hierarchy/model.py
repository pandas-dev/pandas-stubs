"""The operand-hierarchy model: what the checker knows before it reads a stub tree.

The definition tables live here rather than in the checker so the model reads as data,
and so a change to the tier system is a change to one file.

The tier model itself, and the reason tier 1 owns no scanned class, is in the module
docstring of ``check.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# The scanned set is enumerated explicitly. Name heuristics cannot separate forward
# from reflected operations (a forward dunder can itself start with ``__r``, as
# ``__rshift__`` does), and a dunder outside this set — every reflected dunder, like
# ``__radd__``, plus non-operator methods such as ``__init__`` — is not scanned.
FORWARD_BINARY_DUNDERS: Final[frozenset[str]] = frozenset(
    {
        "__add__",
        "__sub__",
        "__mul__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__divmod__",
        "__pow__",
        "__lshift__",
        "__rshift__",
        "__and__",
        "__or__",
        "__xor__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
    }
)

# Operand names that can appear in ``other``, mapped to their tier.
TIER_OPERANDS: Final[dict[str, int]] = {
    "Index": 2,
    "MultiIndex": 2,
    "Series": 3,
    "DataFrame": 4,
}

# Tier 1 -- array-likes such as ``ExtensionArray`` -- is deliberately not registered
# here, and registering a class there is not a one-line addition: a live
# ``"ExtensionArray": 1`` also needs ``core/arrays/base.pyi`` in ``REQUIRED_STUB_FILES``
# (otherwise ``_class_index`` finds no node for the class and the checker fails with
# "could not find class"), a ``1: "ExtensionArray"`` entry in
# ``CANONICAL_OPERAND_BY_TIER`` (otherwise the ``CANONICAL_OPERAND_BY_TIER[...]`` lookup
# in ``check.py`` raises an uncaught ``KeyError``), and an exception key for every tier-0
# scalar whose forward dunders would then name a tier-1 operand. The tier-1 paragraph in
# ``check.py`` states why it is not scanned at all.

# The operand name a tier is registered under, so a site that spells its operand
# ``TimedeltaIndex`` or ``MultiIndex`` is looked up as ``Index``. Keyed by tier, not by
# spelling, so a spelling cannot drift out of the exception list.
CANONICAL_OPERAND_BY_TIER: Final[dict[int, str]] = {
    TIER_OPERANDS["Index"]: "Index",
    TIER_OPERANDS["Series"]: "Series",
    TIER_OPERANDS["DataFrame"]: "DataFrame",
}

# Tier 0 is the scalars. Each is named together with the stub file that declares it, so
# the class scan can require that file rather than silently skipping the class.
TIER_0_CLASSES: Final[dict[str, Path]] = {
    "Timedelta": Path("_libs/tslibs/timedeltas.pyi"),
    "Timestamp": Path("_libs/tslibs/timestamps.pyi"),
    "Period": Path("_libs/tslibs/period.pyi"),
    "Interval": Path("_libs/interval.pyi"),
    "NAType": Path("_libs/missing.pyi"),
}

# Every file that declares a scanned class. The list is explicit so that moving a class
# to a new file fails the scan loudly instead of silently dropping it.
REQUIRED_STUB_FILES: Final[tuple[Path, ...]] = tuple(
    sorted(
        {
            Path("core/base.pyi"),
            Path("core/frame.pyi"),
            Path("core/indexes/base.pyi"),
            Path("core/indexes/category.pyi"),
            Path("core/indexes/datetimelike.pyi"),
            Path("core/indexes/datetimes.pyi"),
            Path("core/indexes/extension.pyi"),
            Path("core/indexes/interval.pyi"),
            Path("core/indexes/multi.pyi"),
            Path("core/indexes/period.pyi"),
            Path("core/indexes/range.pyi"),
            Path("core/indexes/timedeltas.pyi"),
            Path("core/series.pyi"),
            Path("_stubs_only/__init__.pyi"),
            *TIER_0_CLASSES.values(),
        }
    )
)
