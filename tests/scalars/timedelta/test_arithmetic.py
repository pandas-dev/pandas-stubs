"""Test file for arithmetic method on Timedelta objects."""

import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


def test_mul() -> None:
    """Test checking that pd.Timedelta * int / float."""
    a = pd.Timedelta("1 day")
    b = True
    c = 1.0
    d = 5
    e = 1 + 3.0j

    check(assert_type(a * c, pd.Timedelta), pd.Timedelta)
    check(assert_type(c * a, pd.Timedelta), pd.Timedelta)
    check(assert_type(a * d, pd.Timedelta), pd.Timedelta)
    check(assert_type(d * a, pd.Timedelta), pd.Timedelta)

    if TYPE_CHECKING_INVALID_USAGE:
        # pd.Timedelta * bool is not allowed, see GH1418
        _0 = a * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _1 = b * a  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _2 = a * e  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _3 = e * a  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
