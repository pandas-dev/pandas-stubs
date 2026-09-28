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
    "ExtensionArray": 1,
    "Index": 2,
    "MultiIndex": 2,
    "Series": 3,
    "DataFrame": 4,
}

# Tier 1 is the array-likes, represented by the ``ExtensionArray`` ABC. Registering it
# takes three entries -- the operand name above, its canonical name below, and the stub
# file that declares it in ``REQUIRED_STUB_FILES``, which the class scan resolves the
# class through -- and it took no exception key when it landed on 2026-09-28: the only
# classes that can violate against tier 1 are the tier-0 scalars, and none of them names
# an array-like in a forward dunder's ``other``. The first site that does needs a key,
# and it is also the first site to read ``CANONICAL_OPERAND_BY_TIER[1]``, which is why a
# tier may not be registered without one.

# The operand name a tier is registered under, so a site that spells its operand
# ``TimedeltaIndex`` or ``MultiIndex`` is looked up as ``Index``. Keyed by tier, not by
# spelling, so a spelling cannot drift out of the exception list.
CANONICAL_OPERAND_BY_TIER: Final[dict[int, str]] = {
    TIER_OPERANDS["ExtensionArray"]: "ExtensionArray",
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
            Path("_stubs_only/__init__.pyi"),
            *TIER_0_CLASSES.values(),
        }
    )
)
