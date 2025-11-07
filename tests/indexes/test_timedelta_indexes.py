from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing_extensions import (
    assert_type,
)

from tests import (
    check,
    np_1darray_int64,
)


@pytest.fixture
def tdi() -> pd.TimedeltaIndex:
    idx = pd.timedelta_range("1 days", periods=3, freq="D")
    return check(assert_type(idx, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_timedelta_index_properties(tdi: pd.TimedeltaIndex) -> None:
    check(assert_type(tdi.asi8, np_1darray_int64), np_1darray_int64, np.integer)
