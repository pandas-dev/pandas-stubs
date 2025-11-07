from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.core.arrays.timedeltas import TimedeltaArray
import pytest
from typing_extensions import assert_type

from tests import (
    check,
    np_1darray_int64,
)


@pytest.fixture
def arr() -> TimedeltaArray:
    a = pd.array([timedelta(seconds=1), timedelta(seconds=2)])
    return check(assert_type(a, TimedeltaArray), TimedeltaArray, pd.Timedelta)


def test_timedelta_index_properties(arr: TimedeltaArray) -> None:
    check(assert_type(arr.asi8, np_1darray_int64), np_1darray_int64, np.integer)
