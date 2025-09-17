import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

series = pd.DataFrame({"A": [1.0, float("nan"), 2.0]})["A"]


def test_window_any() -> None:
    check(assert_type(series.cumprod(), pd.Series), pd.Series, np.floating)
