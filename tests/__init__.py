from __future__ import annotations

from contextlib import (
    AbstractContextManager,
    nullcontext,
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
PD_LTE_15 = Version(pd.__version__) < Version("1.5.999")


def check(actual: T, klass: type, dtype: type | None = None, attr: str = "left") -> T:
    if not isinstance(actual, klass):
        raise RuntimeError(f"Expected type '{klass}' but got '{type(actual)}'")
    if dtype is None:
        return actual  # type: ignore[return-value]

    if hasattr(actual, "__iter__"):
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

    Notes
    -----
    The lower and upper bounds are exclusive so that a pytest.warns context
    manager is returned if lower < pd.__version__ < upper.

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
    """
    lb = Version("0.0.0") if lower is None else Version(lower)
    ub = Version("9999.0.0") if upper is None else Version(upper)
    current = Version(pd.__version__)
    if lb < current < ub:
        return pytest.warns(warning, match=match)
    else:
        return nullcontext()
