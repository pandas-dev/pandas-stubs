"""Test module for methods related to col function."""

from typing import assert_type

import pandas as pd
from pandas.api.typing import Expression

from tests import check


def test_constructor() -> None:
    """Test constructor for the pd.col."""
    check(assert_type(pd.col("x"), Expression), Expression)


def test_binary_operators() -> None:
    """Test binary operators for Expression."""
    x = pd.col("x")
    y = pd.col("y")

    check(assert_type(x + y, Expression), Expression)
    check(assert_type(x + 1, Expression), Expression)
    check(assert_type(1 + x, Expression), Expression)
    check(assert_type(x - y, Expression), Expression)
    check(assert_type(x - 1, Expression), Expression)
    check(assert_type(1 - x, Expression), Expression)
    check(assert_type(x * y, Expression), Expression)
    check(assert_type(x * 2, Expression), Expression)
    check(assert_type(2 * x, Expression), Expression)
    check(assert_type(x / y, Expression), Expression)
    check(assert_type(x / 2, Expression), Expression)
    check(assert_type(2 / x, Expression), Expression)
    check(assert_type(x // y, Expression), Expression)
    check(assert_type(x // 2, Expression), Expression)
    check(assert_type(2 // x, Expression), Expression)
    check(assert_type(x % y, Expression), Expression)
    check(assert_type(x % 2, Expression), Expression)
    check(assert_type(2 % x, Expression), Expression)
    check(assert_type(x >= y, Expression), Expression)
    check(assert_type(x > y, Expression), Expression)
    check(assert_type(x <= y, Expression), Expression)
    check(assert_type(x < y, Expression), Expression)
    check(assert_type(x == y, Expression), Expression)
    check(assert_type(x != y, Expression), Expression)


def test_logical_operators() -> None:
    """Test logical operators for Expression."""
    x = pd.col("x")
    y = pd.col("y")

    check(assert_type(x & y, Expression), Expression)
    check(assert_type(x & True, Expression), Expression)
    check(assert_type(True & x, Expression), Expression)
    check(assert_type(x | y, Expression), Expression)
    check(assert_type(x | True, Expression), Expression)
    check(assert_type(True | x, Expression), Expression)
    check(assert_type(x ^ y, Expression), Expression)
    check(assert_type(x ^ True, Expression), Expression)
    check(assert_type(True ^ x, Expression), Expression)
    check(assert_type(~x, Expression), Expression)


def test_binary_operators_with_series() -> None:
    """Test binary operators between Expression and Series."""
    x = pd.col("x")
    s = pd.Series([1, 2, 3])

    check(assert_type(x + s, Expression), Expression)
    check(assert_type(x - s, Expression), Expression)
    check(assert_type(x * s, Expression), Expression)
    check(assert_type(x / s, Expression), Expression)
    check(assert_type(x // s, Expression), Expression)
    check(assert_type(x % s, Expression), Expression)
    check(assert_type(x >= s, Expression), Expression)
    check(assert_type(x > s, Expression), Expression)
    check(assert_type(x <= s, Expression), Expression)
    check(assert_type(x < s, Expression), Expression)
    check(assert_type(x == s, Expression), Expression)
    check(assert_type(x != s, Expression), Expression)


def test_logical_operators_with_series() -> None:
    """Test logical operators between Expression and Series."""
    x = pd.col("x")
    s = pd.Series([True, False, True])

    check(assert_type(x & s, Expression), Expression)
    check(assert_type(x | s, Expression), Expression)
    check(assert_type(x ^ s, Expression), Expression)
