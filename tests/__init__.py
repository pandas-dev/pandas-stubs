import typing_extensions
import types


def assert_type(actual_object, expected):
    # rough check whether the types might match

    actual = type(actual_object).__name__.split(".")[-1]
    error = False

    if isinstance(expected, types.GenericAlias):  # type: ignore[attr-defined]
        expected = expected.__name__

    if isinstance(expected, str):
        actual = actual.lower()
        expected = expected.lower()
        error = actual not in expected
    elif expected is None:
        error = actual_object is not None
    else:
        error = not isinstance(actual_object, expected)

    if error:
        raise TypeError(f"Expected '{expected}' got '{actual}'")


typing_extensions.assert_type = assert_type
