from typing_extensions import assert_type
import numpy as np
import pandas as pd

from numpy import typing as npt


def test_index_unique():

    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=pd.Index([1, 2, 3, 2]))
    ind = df.index
    assert_type(ind, "pd.Index")
    i2 = ind.unique()
    assert_type(i2, "pd.Index")


def test_index_isin():
    ind = pd.Index([1, 2, 3, 4, 5])
    isin = ind.isin([2, 4])
    assert_type(isin, "npt.NDArray[np.bool_]")


def test_index_astype():
    indi = pd.Index([1, 2, 3])
    inds = pd.Index(["a", "b", "c"])
    indc = indi.astype(inds.dtype)
    assert_type(indc, "pd.Index")
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    mia = mi.astype(object)  # object is only valid parameter for MultiIndex.astype()
    assert_type(mia, "pd.MultiIndex")


def test_multiindex_get_level_values():
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    i1 = mi.get_level_values("ab")
    assert_type(i1, "pd.Index")


def test_index_tolist() -> None:
    i1 = pd.Index([1, 2, 3])
    l1 = i1.tolist()
    i2 = i1.to_list()
