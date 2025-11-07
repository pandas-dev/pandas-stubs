from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.arrays.datetimes import DatetimeArray
import pytest
from typing_extensions import assert_type

from tests import (
    check,
    np_1darray_int64,
)


@pytest.fixture
def arr() -> DatetimeArray:
    a = pd.array([datetime(2025, 11, 7), datetime(2025, 11, 8)])
    return check(assert_type(a, DatetimeArray), DatetimeArray, pd.Timestamp)


def test_timedelta_index_properties(arr: DatetimeArray) -> None:
    check(assert_type(arr.asi8, np_1darray_int64), np_1darray_int64, np.integer)
