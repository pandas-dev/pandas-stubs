from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd

from tests import check


def test_relops_ordered_categorical() -> None:
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True))
    other = left.copy()

    check(assert_type(left < "b", "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= "b", "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > "b", "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= "b", "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)
