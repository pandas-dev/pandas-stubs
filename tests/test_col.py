"""Test module for methods related to col function."""

from typing import assert_type

import pandas as pd
from pandas.api.typing import Expression

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


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


def test_str_accessor() -> None:
    """Test the str accessor for Expression."""
    df = pd.DataFrame({"name": ["beluga", "narwhal"], "speed": [100, 110]})
    check(
        assert_type(df.assign(name_titlecase=pd.col("name").str.title()), pd.DataFrame),
        pd.DataFrame,
    )


def test_indexing() -> None:
    """Test DataFrame indexing with Expression."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    check(assert_type(df.loc[pd.col("a") > 1], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[pd.col("a") > 1], pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.loc[(pd.col("a") > 1) & (pd.col("b") < 6.0)], pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.loc[pd.col("a") > 1, "b"], pd.Series), pd.Series, float)
    check(assert_type(df.loc[pd.col("a") > 1, ["a", "b"]], pd.DataFrame), pd.DataFrame)

    df.loc[pd.col("a") > 1] = 0

    # `Series` has no columns, so expression indexing fails at runtime
    if TYPE_CHECKING_INVALID_USAGE:
        s = df["a"]
        _0 = s.loc[pd.col("a") > 1]  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue,reportUnknownVariableType] # ty: ignore[invalid-argument-type] # pyrefly: ignore[bad-index]
        _1 = s[pd.col("a") > 1]  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue,reportUnknownVariableType] # ty: ignore[invalid-argument-type] # pyrefly: ignore[bad-index]
