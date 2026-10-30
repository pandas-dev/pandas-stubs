from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd

from tests import check


def test_types_element_wise_arithmetic() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df2 = pd.DataFrame(data={"col1": [10, 20], "col3": [3, 4]})

    check(assert_type(df + df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.add(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df - df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.sub(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df * df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mul(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df / df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.div(df2, fill_value=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(df / [2, 2], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.div([2, 2], fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df // df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.floordiv(df2, fill_value=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(df // [2, 2], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.floordiv([2, 2], fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df % df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mod(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df2**df, pd.DataFrame), pd.DataFrame)
    check(assert_type(df2.pow(df, fill_value=0), pd.DataFrame), pd.DataFrame)

    # divmod operation was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    check(
        assert_type(divmod(df, df2), tuple[pd.DataFrame, pd.DataFrame]),
        tuple,
        pd.DataFrame,
    )
    check(
        assert_type(df.__divmod__(df2), tuple[pd.DataFrame, pd.DataFrame]),
        tuple,
        pd.DataFrame,
    )
    check(
        assert_type(df.__rdivmod__(df2), tuple[pd.DataFrame, pd.DataFrame]),
        tuple,
        pd.DataFrame,
    )


def test_types_scalar_arithmetic() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})

    check(assert_type(df + 1, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.add(1, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df - 1, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.sub(1, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df * 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mul(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df / 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.div(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df // 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.floordiv(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df % 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mod(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df**2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df**0, pd.DataFrame), pd.DataFrame)
    check(assert_type(df**0.213, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.pow(0.5), pd.DataFrame), pd.DataFrame)


def test_types_dot() -> None:
    df1 = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
    df2 = pd.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])
    s1 = pd.Series([1, 1, 2, 1])
    np_array = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
    check(assert_type(df1 @ df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.dot(df2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1 @ np_array, pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.dot(np_array), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1 @ s1, pd.Series), pd.Series)
    check(assert_type(s1 @ df2, pd.Series), pd.Series)
    check(assert_type(df1.dot(s1), pd.Series), pd.Series)
    list_like = [[0, 1], [1, 2], [-1, -1], [2, 0]]
    check(assert_type(df1.dot(list_like), pd.DataFrame), pd.DataFrame)
    time = pd.DataFrame({"HH": [1, 2, 3], "MM": [4, 5, 6]})
    seconds = time.dot([3600, 60])
    check(assert_type(seconds, pd.Series), pd.Series)
