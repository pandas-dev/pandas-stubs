from __future__ import annotations

from typing import Callable


def check(actual: object, klass: type, dtype: type | None = None) -> None:
    if not isinstance(actual, klass):
        raise TypeError
    if dtype is None:
        return None

    if hasattr(actual, "__iter__"):
        value = next(iter(actual))  # type: ignore[call-overload]
    else:
        value = actual.left  # type: ignore[attr-defined]

    if not isinstance(value, dtype):
        raise TypeError
