from __future__ import annotations

from collections.abc import (
    Generator,
    Iterable,
)
from contextlib import (
    AbstractContextManager,
    nullcontext,
    suppress,
)
from datetime import timezone
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    cast,
    get_args,
    get_origin,
)

import numpy as np
import pandas as pd

# Next set of imports is to keep the private imports needed for testing
# in one place
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.util.version import Version
import pytest

from pandas.core.dtypes.base import ExtensionDtype

if TYPE_CHECKING:
    from pandas._typing import T

TYPE_CHECKING_INVALID_USAGE: Final = TYPE_CHECKING
LINUX = sys.platform == "linux"
WINDOWS = sys.platform in {"win32", "cygwin"}
MAC = sys.platform == "darwin"
PD_LTE_23 = Version(pd.__version__) < Version("2.3.999")
NUMPY20 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"


def check(
    actual: T,
    klass: type,
    dtype: type | None = None,
    attr: str = "left",
    index_to_check_for_type: Literal[0, -1] = 0,
) -> T:
    __tracebackhide__ = True
    origin = get_origin(klass)
    if not isinstance(actual, origin or klass):
        raise RuntimeError(f"Expected type '{klass}' but got '{type(actual)}'")
    if origin is np.ndarray:
        # Check shape and dtype
        args = get_args(klass)
        shape_type = args[0] if len(args) >= 1 else None
        dtype_type = args[1] if len(args) >= 2 else None
        if (
            shape_type
            and get_origin(shape_type) is tuple
            and (tuple_args := get_args(shape_type))
            and ... not in tuple_args  # fixed-length tuple
            and (arr_ndim := getattr(actual, "ndim"))  # noqa: B009
            != (expected_ndim := len(tuple_args))
        ):
            raise RuntimeError(
                f"Array has wrong dimension {arr_ndim}, expected {expected_ndim}"
            )

        if (
            dtype_type
            and get_origin(dtype_type) is np.dtype
            and (dtype_args := get_args(dtype_type))
            and isinstance((expected_dtype := dtype_args[0]), type)
            and issubclass(expected_dtype, np.generic)
            and (arr_dtype := getattr(actual, "dtype")) != expected_dtype  # noqa: B009
        ):
            raise RuntimeError(
                f"Array has wrong dtype {arr_dtype}, expected {expected_dtype.__name__}"
            )

    if dtype is None:
        return actual

    value: Any
    if isinstance(actual, pd.Series):
        # pyright ignore is by design microsoft/pyright#11191
        value = cast(pd.Series, actual).iloc[index_to_check_for_type]
    elif isinstance(actual, pd.Index):
        # pyright ignore is by design microsoft/pyright#11191
        value = cast(pd.Index, actual)[index_to_check_for_type]
    elif isinstance(actual, BaseGroupBy):
        # `BaseGroupBy.obj` is internal and untyped
        value = actual.obj  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue,reportUnknownVariableType]
    elif isinstance(actual, Iterable):
        # T_co in Iterable[T_co] does not have a default value and `actual` is Iterable[Unknown] by pyright
        value = next(iter(cast("Iterable[Any]", actual)))
    else:
        assert hasattr(actual, attr)
        value = getattr(actual, attr)

    if not isinstance(value, dtype):
        # pyright ignore is by design microsoft/pyright#11191
        raise RuntimeError(
            f"Expected type '{dtype}' but got '{type(value)}'"  # pyright: ignore[reportUnknownArgumentType]
        )
    # pyright ignore is by design microsoft/pyright#11190
    return actual  # pyright: ignore[reportUnknownVariableType]


def pytest_warns_bounded(
    warning: type[Warning],
    match: str,
    lower: str | None = None,
    upper: str | None = None,
    version_str: str | None = None,
    upper_exception: type[Exception] | None = None,
) -> AbstractContextManager[Any]:
    """
    Version conditional pytest.warns context manager

    Returns a context manager that will raise an error if
    the warning is not issued when pandas version is
    between the lower and upper version given.

    Parameters
    ----------
    warning : type[Warning]
        The warning class to check for.
    match : str
        The string to match in the warning message.
    lower : str, optional
        The lower bound of the version to check for the warning.
    upper : str, optional
        The upper bound of the version to check for the warning.
    version_str: str, optional
        The version string to use.  If None, then uses the pandas version.
        Can be used to check a python version as well
    upper_exception: Exception, optional
        Exception to catch if the pandas version is greater than or equal to
        the upper bound

    Notes
    -----
    The lower and upper bounds are exclusive so that a pytest.warns context
    manager is returned if lower < version_str < upper.

    Examples
    --------
    with pytest_warns_bounded(UserWarning, match="foo", lower="1.2.99"):
        # Versions 1.3.0 and above will raise an error
        # if the warning is not issued
        pass

    with pytest_warns_bounded(UserWarning, match="foo", upper="1.5.99"):
        # Versions 1.6.0 and below will raise an error
        # if the warning is not issued
        pass

    with pytest_warns_bounded(
        UserWarning, match="foo", lower="1.2.99", upper="1.5.99"
    ):
        # Versions between 1.3.x and 1.5.x will raise an error
        pass

    with pytest_warns_bounded(
        UserWarning, match="foo", lower="3.10",
        version_str = platform.python_version()
    ):
        # Python version 3.11 and above will raise an error
        # if the warning is not issued
        pass

    with pytest_warns_bounded(
        UserWarning, match="foo", lower="1.2.99", upper="1.5.99",
        upper_exception=AttributeError
    ):
        # Versions between 1.3.x and 1.5.x will raise an error
        # Above 1.5.x, we expect an `AttributeError` to be raised
        pass

    """
    lb = Version("0.0.0") if lower is None else Version(lower)
    ub = Version("9999.0.0") if upper is None else Version(upper)
    if version_str is None:
        current = Version(pd.__version__)
    else:
        current = Version(version_str)
    if lb < current < ub:
        return pytest.warns(warning, match=match)
    if upper_exception is None:
        return nullcontext()
    return suppress(upper_exception)


def exception_on_platform(dtype: type | str | ExtensionDtype) -> type[Exception] | None:
    if (WINDOWS or MAC) and dtype in {"f16", "float128", "c32", "complex256"}:
        return TypeError
    return None


def get_dtype(dtype: object) -> Generator[Any, None, None]:
    """Extract types and string literals from a Union or Literal type."""
    if isinstance(dtype, str):
        yield dtype
    # isinstance(type[bool], type) is True in py310, but not in newer versions
    elif isinstance(dtype, type) and not str(dtype).startswith("type["):
        if dtype is pd.DatetimeTZDtype:
            yield dtype(tz=timezone.utc)
        elif "pandas" in str(dtype):
            yield dtype()
        else:
            yield dtype
    else:
        for arg in get_args(dtype):
            yield from get_dtype(arg)
