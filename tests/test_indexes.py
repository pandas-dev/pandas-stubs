import numpy as np
from numpy import typing as npt
import pandas as pd
from typing_extensions import assert_type


def test_index_unique() -> None:

    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=pd.Index([1, 2, 3, 2]))
    ind = df.index
    assert isinstance(assert_type(ind, "pd.Index"), pd.Index)
    i2 = ind.unique()
    assert isinstance(assert_type(i2, "pd.Index"), pd.Index)


def test_index_isin() -> None:
    ind = pd.Index([1, 2, 3, 4, 5])
    isin = ind.isin([2, 4])
    assert isinstance(assert_type(isin, "npt.NDArray[np.bool_]"), np.ndarray)


def test_index_astype() -> None:
    indi = pd.Index([1, 2, 3])
    inds = pd.Index(["a", "b", "c"])
    indc = indi.astype(inds.dtype)
    assert isinstance(assert_type(indc, "pd.Index"), pd.Index)
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    mia = mi.astype(object)  # object is only valid parameter for MultiIndex.astype()
    assert isinstance(assert_type(mia, "pd.MultiIndex"), pd.MultiIndex)


def test_multiindex_get_level_values() -> None:
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    i1 = mi.get_level_values("ab")
    assert isinstance(assert_type(i1, "pd.Index"), pd.Index)


def test_index_tolist() -> None:
    i1 = pd.Index([1, 2, 3])
    l1 = i1.tolist()
    i2 = i1.to_list()


def test_column_getitem() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199#issuecomment-1132806594
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    column = df.columns[0]
    a = df[column]


def test_column_contains() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199
    df = pd.DataFrame({"A": [1, 2], "B": ["c", "d"], "E": [3, 4]})

    collist = [column for column in df.columns]

    collist2 = [column for column in df.columns[df.columns.str.contains("A|B")]]

    length = len(df.columns[df.columns.str.contains("A|B")])


def test_difference_none() -> None:
    # https://github.com/pandas-dev/pandas-stubs/issues/17
    ind = pd.Index([1, 2, 3])
    id = ind.difference([1, None])
