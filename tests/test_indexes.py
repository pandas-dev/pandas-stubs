from __future__ import annotations

import numpy as np
from numpy import typing as npt
import pandas as pd
from pandas.core.indexes.numeric import NumericIndex
from typing_extensions import assert_type

from tests import check


def test_index_unique() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=pd.Index([1, 2, 3, 2]))
    ind = df.index
    check(assert_type(ind, pd.Index), pd.Index)
    i2 = ind.unique()
    check(assert_type(i2, pd.Index), pd.Index)


def test_index_isin() -> None:
    ind = pd.Index([1, 2, 3, 4, 5])
    isin = ind.isin([2, 4])
    check(assert_type(isin, npt.NDArray[np.bool_]), np.ndarray, np.bool_)


def test_index_astype() -> None:
    indi = pd.Index([1, 2, 3])
    inds = pd.Index(["a", "b", "c"])
    indc = indi.astype(inds.dtype)
    check(assert_type(indc, pd.Index), pd.Index)
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    mia = mi.astype(object)  # object is only valid parameter for MultiIndex.astype()
    check(assert_type(mia, pd.MultiIndex), pd.MultiIndex)


def test_multiindex_get_level_values() -> None:
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    i1 = mi.get_level_values("ab")
    check(assert_type(i1, pd.Index), pd.Index)


def test_index_tolist() -> None:
    i1 = pd.Index([1, 2, 3])
    check(assert_type(i1.tolist(), list), list, int)
    check(assert_type(i1.to_list(), list), list, int)


def test_column_getitem() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199#issuecomment-1132806594
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    column = df.columns[0]
    check(assert_type(df[column], pd.Series), pd.Series, int)


def test_column_contains() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199
    df = pd.DataFrame({"A": [1, 2], "B": ["c", "d"], "E": [3, 4]})

    collist = [column for column in df.columns]

    collist2 = [column for column in df.columns[df.columns.str.contains("A|B")]]

    length = len(df.columns[df.columns.str.contains("A|B")])


def test_difference_none() -> None:
    # https://github.com/pandas-dev/pandas-stubs/issues/17
    ind = pd.Index([1, 2, 3])
    check(assert_type(ind.difference([1, None]), pd.Index), pd.Index)
    # GH 253
    check(assert_type(ind.difference([1]), pd.Index), pd.Index)


def test_str_split() -> None:
    # GH 194
    ind = pd.Index(["a-b", "c-d"])
    check(assert_type(ind.str.split("-"), pd.Index), pd.Index)
    check(assert_type(ind.str.split("-", expand=True), pd.MultiIndex), pd.MultiIndex)


def test_index_dropna():
    idx = pd.Index([1, 2])

    check(assert_type(idx.dropna(how="all"), pd.Index), pd.Index)
    check(assert_type(idx.dropna(how="any"), pd.Index), pd.Index)

    midx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]])

    check(assert_type(midx.dropna(how="all"), pd.MultiIndex), pd.MultiIndex)
    check(assert_type(midx.dropna(how="any"), pd.MultiIndex), pd.MultiIndex)


def test_index_neg():
    # GH 253
    idx = pd.Index([1, 2])
    check(assert_type(-idx, pd.Index), pd.Index)


def test_types_to_numpy() -> None:
    idx = pd.Index([1, 2])
    check(assert_type(idx.to_numpy(), np.ndarray), np.ndarray)
    check(assert_type(idx.to_numpy(dtype="int", copy=True), np.ndarray), np.ndarray)
    check(assert_type(idx.to_numpy(na_value=0), np.ndarray), np.ndarray)


def test_index_arithmetic() -> None:
    # GH 287
    idx = pd.Index([1, 2.2, 3], dtype=float)
    check(assert_type(idx + 3, NumericIndex), NumericIndex)
    check(assert_type(idx - 3, NumericIndex), NumericIndex)
    check(assert_type(idx * 3, NumericIndex), NumericIndex)
    check(assert_type(idx / 3, NumericIndex), NumericIndex)
    check(assert_type(idx // 3, NumericIndex), NumericIndex)
    check(assert_type(3 + idx, NumericIndex), NumericIndex)
    check(assert_type(3 - idx, NumericIndex), NumericIndex)
    check(assert_type(3 * idx, NumericIndex), NumericIndex)
    check(assert_type(3 / idx, NumericIndex), NumericIndex)
    check(assert_type(3 // idx, NumericIndex), NumericIndex)
