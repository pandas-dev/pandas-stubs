from typing import assert_type

import numpy as np
import pandas as pd

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import np_1darray_intp


def test_period_index_shift() -> None:
    ind = pd.period_range(start="2022-06-01", periods=10)
    check(assert_type(ind.shift(1), pd.PeriodIndex), pd.PeriodIndex)


def test_period_index_constructor() -> None:
    check(
        assert_type(pd.PeriodIndex(["2000"], dtype="period[D]"), pd.PeriodIndex),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.PeriodIndex(["2000"], freq="D", name="foo", copy=True), pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )


def test_period_index_from_ordinals() -> None:
    check(
        assert_type(pd.PeriodIndex.from_ordinals([0, 1], freq="D"), pd.PeriodIndex),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.PeriodIndex.from_ordinals(np.array([0, 1]), freq="D", name="foo"),
            pd.PeriodIndex,
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.PeriodIndex.from_ordinals(pd.Index([0, 1]), freq="D"), pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(pd.PeriodIndex.from_ordinals(range(2), freq="D"), pd.PeriodIndex),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.PeriodIndex.from_ordinals(pd.Series([0, 1]), freq="D"), pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.PeriodIndex.from_ordinals(pd.array([0, 1]), freq="D"), pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        pd.PeriodIndex.from_ordinals([0, 1])  # type: ignore[call-arg] # pyright: ignore[reportCallIssue] # pyrefly: ignore[missing-argument] # ty: ignore[missing-argument]


def test_period_index_asof_locs() -> None:
    idx = pd.PeriodIndex(["2000", "2001"], freq="D")
    where = pd.DatetimeIndex(["2023-05-30 00:12:00", "2023-06-01 00:00:00"])
    mask = np.ones(2, dtype=bool)
    check(assert_type(idx.asof_locs(where, mask), np_1darray_intp), np_1darray_intp)
