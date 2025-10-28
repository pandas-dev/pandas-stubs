from __future__ import annotations

import pandas as pd
from typing_extensions import (
    assert_type,
)

from tests import (
    check,
)


def test_rangeindex_floordiv() -> None:
    ri = pd.RangeIndex(3)
    check(
        assert_type(ri // 2, "pd.Index[int]"),
        pd.Index,
    )
