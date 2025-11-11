from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.core.arrays.timedeltas import TimedeltaArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    td = timedelta(2025, 11, 10)
    np_dt = np.timedelta64(td)
    check(assert_type(pd.array([td]), TimedeltaArray), TimedeltaArray)
    check(
        assert_type(pd.array([td, pd.Timedelta(td), np_dt]), TimedeltaArray),
        TimedeltaArray,
    )
    check(assert_type(pd.array([td, None]), TimedeltaArray), TimedeltaArray)
    check(assert_type(pd.array([td, pd.NaT, None]), TimedeltaArray), TimedeltaArray)

    check(
        assert_type(  # type: ignore[assert-type]  # I do not understand
            pd.array(np.array([td], np.timedelta64)), TimedeltaArray
        ),
        TimedeltaArray,
    )

    check(assert_type(pd.array(pd.array([td])), TimedeltaArray), TimedeltaArray)

    check(assert_type(pd.array(pd.Index([td])), TimedeltaArray), TimedeltaArray)

    check(assert_type(pd.array(pd.Series([td])), TimedeltaArray), TimedeltaArray)
