from __future__ import annotations

from pandas._typing import T


def check(actual: T, klass: type, dtype: type | None = None, attr: str = "left") -> T:

    if not isinstance(actual, klass):
        raise RuntimeError(f"Expected type '{klass}' but got '{type(actual)}'")
    if dtype is None:
        return actual

    if hasattr(actual, "__iter__"):
        value = next(iter(actual))  # type: ignore[call-overload]
    else:
        assert hasattr(actual, attr)
        value = getattr(actual, attr)

    if not isinstance(value, dtype):
        raise RuntimeError(f"Expected type '{dtype}' but got '{type(value)}'")
    return actual
