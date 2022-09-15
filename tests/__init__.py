from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Final,
)

from packaging.version import parse
import pandas as pd

from pandas._typing import T

TYPE_CHECKING_INVALID_USAGE: Final = TYPE_CHECKING
PD_LT_15 = parse(pd.__version__) < parse("1.5.0")


def check(actual: T, klass: type, dtype: type | None = None, attr: str = "left") -> T:
    if not isinstance(actual, klass):
        raise RuntimeError(f"Expected type '{klass}' but got '{type(actual)}'")
    if dtype is None:
        return actual  # type: ignore[return-value]

    if hasattr(actual, "__iter__"):
        value = next(iter(actual))  # type: ignore[call-overload]
    else:
        assert hasattr(actual, attr)
        value = getattr(actual, attr)

    if not isinstance(value, dtype):
        raise RuntimeError(f"Expected type '{dtype}' but got '{type(value)}'")
    return actual  # type: ignore[return-value]
