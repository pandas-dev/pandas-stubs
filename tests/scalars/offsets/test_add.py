import datetime as dt
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType

from tests import check

from pandas.tseries.offsets import Easter

# No anchor offset declares `__add__`/`__radd__` on this branch, so they all resolve through
# the inherited `BaseOffset` signature; naming the concrete `Easter` receiver is what makes
# each `assert_type` exact.  Durations and other offsets are rejected at runtime but still
# type-check here -- pandas-dev/pandas-stubs#1943 removes those overloads and asserts the
# rejection in `easter/test_add.py`.


def test_easter_python_scalars() -> None:
    """`Easter` maps `date`/`datetime` operands to a `Timestamp` in both directions."""
    e = Easter()
    check(assert_type(e + dt.date(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.date(2026, 10, 2) + e, pd.Timestamp), pd.Timestamp)
    check(assert_type(e + dt.datetime(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 10, 2) + e, pd.Timestamp), pd.Timestamp)


def test_easter_numpy_scalars() -> None:
    """`Easter` maps a `np.datetime64` operand to a `Timestamp` in both directions."""
    e = Easter()
    check(assert_type(e + np.datetime64("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(np.datetime64("2026-10-02") + e, pd.Timestamp), pd.Timestamp)


def test_easter_pandas_scalars() -> None:
    """`Easter` maps a `Timestamp` operand to a `Timestamp` and propagates `NaT`."""
    e = Easter()
    check(assert_type(e + pd.Timestamp("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp("2026-10-02") + e, pd.Timestamp), pd.Timestamp)
    check(assert_type(e + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + e, NaTType), NaTType)
