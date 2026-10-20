"""The operand-hierarchy model: what the checker knows before it reads a stub tree.

The definition tables live here rather than in the checker so the model reads as data,
and so a change to the tier system is a change to one file.

The tier model itself is documented in ``docs/type-architecture/operand-hierarchy.md``.
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

# Every scanned class and its tier; Index subclasses are discovered through their
# base chain.
CLASS_TIERS: Final[dict[str, int]] = {
    "Timedelta": 0,
    "Timestamp": 0,
    "Period": 0,
    "Interval": 0,
    "NAType": 0,
    "ExtensionArray": 1,
    "Index": 2,
    "MultiIndex": 2,
    "Series": 3,
    "DataFrame": 4,
}

# Index subclass spellings share the canonical operand Index.
CANONICAL_OPERAND_BY_TIER: Final[dict[int, str]] = {
    CLASS_TIERS["ExtensionArray"]: "ExtensionArray",
    CLASS_TIERS["Index"]: "Index",
    CLASS_TIERS["Series"]: "Series",
    CLASS_TIERS["DataFrame"]: "DataFrame",
}

# Every file that declares a scanned class. The list is explicit so that moving a class
# to a new file fails the scan loudly instead of silently dropping it.
REQUIRED_STUB_FILES: Final[tuple[Path, ...]] = (
    Path("_libs/interval.pyi"),
    Path("_libs/missing.pyi"),
    Path("_libs/tslibs/period.pyi"),
    Path("_libs/tslibs/timedeltas.pyi"),
    Path("_libs/tslibs/timestamps.pyi"),
    Path("_stubs_only/__init__.pyi"),
    Path("core/arrays/base.pyi"),
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
)
