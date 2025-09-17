import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

series = pd.Series([3j, 3 + 4j, 2j])


def test_window_complex() -> None:
    check(
        assert_type(series.cumprod(), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
