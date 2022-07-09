from __future__ import annotations


def check(
    actual: object, klass: type, dtype: type | None = None, attr: str = "left"
) -> None:

    if not isinstance(actual, klass):
        raise RuntimeError(f"Expected type '{klass}' but got '{type(actual)}'")
    if dtype is None:
        return None

    if hasattr(actual, "__iter__"):
        value = next(iter(actual))  # type: ignore[call-overload]
    else:
        assert hasattr(actual, attr)
        value = getattr(actual, attr)  # type: ignore[attr-defined]

    if not isinstance(value, dtype):
        raise RuntimeError(f"Expected type '{dtype}' but got '{type(value)}'")
    return None
