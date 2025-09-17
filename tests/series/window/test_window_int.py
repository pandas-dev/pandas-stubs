import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

series = pd.Series([3, 1, 2])


def test_window_int() -> None:
    check(assert_type(series.cumprod(), "pd.Series[int]"), pd.Series, np.integer)
