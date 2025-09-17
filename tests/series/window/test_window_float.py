import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

series = pd.Series([3.0, float("nan"), 2.0])


def test_window_float() -> None:
    check(assert_type(series.cumprod(), "pd.Series[float]"), pd.Series, np.floating)
