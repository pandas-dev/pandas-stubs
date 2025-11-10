from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.arrays.datetimes import DatetimeArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    dt = datetime(2025, 11, 10)
    check(assert_type(pd.array([dt]), DatetimeArray), DatetimeArray)
    check(assert_type(pd.array([dt, pd.Timestamp(dt)]), DatetimeArray), DatetimeArray)
    check(assert_type(pd.array([dt, None]), DatetimeArray), DatetimeArray)
    check(assert_type(pd.array([dt, pd.NaT, None]), DatetimeArray), DatetimeArray)

    np_dt = np.datetime64(dt)
    check(assert_type(pd.array([np_dt]), DatetimeArray), DatetimeArray)
    check(assert_type(pd.array([np_dt, None]), DatetimeArray), DatetimeArray)
    check(assert_type(pd.array([np_dt, pd.NaT]), DatetimeArray), DatetimeArray)

    check(
        assert_type(  # type: ignore[assert-type]  # I do not understand
            pd.array(np.array([dt], np.datetime64)), DatetimeArray
        ),
        DatetimeArray,
    )

    check(assert_type(pd.array(pd.array([dt])), DatetimeArray), DatetimeArray)

    check(assert_type(pd.array(pd.Index([dt])), DatetimeArray), DatetimeArray)

    check(assert_type(pd.array(pd.Series([dt])), DatetimeArray), DatetimeArray)
