from __future__ import annotations

from contextlib import (
    AbstractContextManager,
    nullcontext,
    suppress,
)
import os
import platform
from typing import (
    TYPE_CHECKING,
    Final,
)

import pandas as pd
from pandas.util.version import Version
import pytest

from pandas._typing import T

TYPE_CHECKING_INVALID_USAGE: Final = TYPE_CHECKING
WINDOWS = os.name == "nt" or "cygwin" in platform.system().lower()
PD_LTE_20 = Version(pd.__version__) < Version("2.0.999")


def check(actual: T, klass: type, dtype: type | None = None, attr: str = "left") -> T:
    if not isinstance(actual, klass):
        raise RuntimeError(f"Expected type '{klass}' but got '{type(actual)}'")
    if dtype is None:
        return actual  # type: ignore[return-value]

    if isinstance(actual, pd.Series):
        value = actual.iloc[0]
    elif isinstance(actual, pd.Index):
        value = actual[0]  # type: ignore[assignment]
    elif hasattr(actual, "__iter__"):
        value = next(iter(actual))  # pyright: ignore[reportGeneralTypeIssues]
    else:
        assert hasattr(actual, attr)
        value = getattr(actual, attr)

    if not isinstance(value, dtype):
        raise RuntimeError(f"Expected type '{dtype}' but got '{type(value)}'")
    return actual  # type: ignore[return-value]


def pytest_warns_bounded(
    warning: type[Warning],
    match: str,
    lower: str | None = None,
    upper: str | None = None,
    version_str: str | None = None,
    upper_exception: type[Exception] | None = None,
) -> AbstractContextManager:
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
    else:
        if upper_exception is None:
            return nullcontext()
        else:
            return suppress(upper_exception)
