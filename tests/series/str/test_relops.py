from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left() -> pd.Series[str]:
    """Left operand"""
    lo = pd.Series(["1", "2", "3"])
    return check(assert_type(lo, "pd.Series[str]"), pd.Series, str)


def test_relops_py_scalar(left: pd.Series[str]) -> None:
    """Test pd.Series[str] <comparison> py scalar"""
    # GH 175
    s: str = "abc"

    check(assert_type(left > s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= s, "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid(left: pd.Series[str]) -> None:
    """Test invalid pd.Series[str] comparisons"""
    f: float = 3.14

    if TYPE_CHECKING_INVALID_USAGE:
        _lt_f = left < f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _gt_f = left > f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _le_f = left <= f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _ge_f = left >= f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
