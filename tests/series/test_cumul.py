import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


def test_cumul_any_float() -> None:
    series = pd.DataFrame({"A": [1.0, float("nan"), 2.0]})["A"]
    check(assert_type(series.cumprod(), pd.Series), pd.Series, np.floating)


def test_cumul_bool() -> None:
    series = pd.Series([True, False, True])
    check(assert_type(series.cumprod(), "pd.Series[int]"), pd.Series, np.integer)


def test_cumul_int() -> None:
    series = pd.Series([3, 1, 2])
    check(assert_type(series.cumprod(), "pd.Series[int]"), pd.Series, np.integer)


def test_cumul_float() -> None:
    series = pd.Series([3.0, float("nan"), 2.0])
    check(assert_type(series.cumprod(), "pd.Series[float]"), pd.Series, np.floating)


def test_cumul_complex() -> None:
    series = pd.Series([3j, 3 + 4j, 2j])
    check(
        assert_type(series.cumprod(), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )


def test_cumul_str() -> None:
    series = pd.Series(["1", "a", "🐼"])
    if TYPE_CHECKING_INVALID_USAGE:
        series.cumprod()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]


def test_cumul_ts() -> None:
    series = pd.Series(pd.to_datetime(["2025-09-18", "2025-09-18", "2025-09-18"]))
    check(assert_type(series, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        series.cumprod()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]
