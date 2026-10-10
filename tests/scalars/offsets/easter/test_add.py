import datetime as dt
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import check

from pandas.tseries.offsets import Easter


@pytest.fixture
def left() -> Easter:
    """The Easter offset under test."""
    return Easter()


def test_easter_python_scalars(left: Easter) -> None:
    """`Easter` maps `date`/`datetime` operands to a `Timestamp` in both directions."""
    check(assert_type(left + dt.date(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.date(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.datetime(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)


def test_easter_numpy_scalars(left: Easter) -> None:
    """`Easter` maps a `np.datetime64` operand to a `Timestamp` in both directions."""
    check(assert_type(left + np.datetime64("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(np.datetime64("2026-10-02") + left, pd.Timestamp), pd.Timestamp)


def test_easter_pandas_scalars(left: Easter) -> None:
    """`Easter` maps a `Timestamp` operand to a `Timestamp` and propagates `NaT`."""
    check(assert_type(left + pd.Timestamp("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)
