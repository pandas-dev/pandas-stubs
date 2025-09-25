from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_bounded,
)


def test_agg_any_float() -> None:
    series = pd.DataFrame({"A": [1.0, float("nan"), 2.0]})["A"]
    check(assert_type(series.median(), Any), np.float64)


def test_agg_bool() -> None:
    series = pd.Series([True, False, True])
    check(assert_type(series.median(), float), np.float64)


def test_agg_int() -> None:
    series = pd.Series([3, 1, 2])
    check(assert_type(series.median(), float), np.float64)


def test_agg_float() -> None:
    series = pd.Series([3.0, float("nan"), 2.0])
    check(assert_type(series.median(), float), np.float64)


def test_agg_complex() -> None:
    series = pd.Series([3j, 3 + 4j, 2j])
    with pytest_warns_bounded(
        np.exceptions.ComplexWarning,
        r"Casting complex values to real discards the imaginary part",
    ):
        check(assert_type(series.median(), float), np.float64)


def test_agg_str() -> None:
    series = pd.Series(["1", "a", "🐼"])
    if TYPE_CHECKING_INVALID_USAGE:
        series.median()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]


def test_agg_ts() -> None:
    series = pd.Series(pd.to_datetime(["2025-09-18", "2025-09-18", "2025-09-18"]))
    check(assert_type(series, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(series.median(), pd.Timestamp), pd.Timestamp)


def test_agg_td() -> None:
    series = pd.Series(pd.to_timedelta(["1 days", "2 days", "3 days"]))
    check(assert_type(series, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(series.median(), pd.Timedelta), pd.Timedelta)
