"""Test file for arithmetic method on Timedelta objects."""

import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


def test_mul_bool() -> None:
    """Test checking that pd.Timedelta * bool is not allowed GH1418."""
    a = pd.Timedelta("1 day")
    b = True
    c = 1.0
    d = 5

    check(assert_type(a * c, pd.Timedelta), pd.Timedelta)
    check(assert_type(a * d, pd.Timedelta), pd.Timedelta)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = a * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _1 = b * a  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
