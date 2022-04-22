import numpy as np
import pandas as pd


from . import check_index_result, check_multiindex_result, check_numpy_result


def test_index_unique():

    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=pd.Index([1, 2, 3, 2]))
    ind = df.index
    i2 = ind.unique()
    check_index_result(i2)


def test_index_isin():
    ind = pd.Index([1, 2, 3, 4, 5])
    isin = ind.isin([2, 4])
    check_numpy_result(isin, np.bool_)


def test_index_astype():
    indi = pd.Index([1, 2, 3])
    inds = pd.Index(["a", "b", "c"])
    indc = indi.astype(inds.dtype)
    check_index_result(indc)
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    mia = mi.astype(object)  # object is only valid parameter for MultiIndex.astype()
    check_multiindex_result(mia)


def test_multiindex_get_level_values():
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    i1 = mi.get_level_values("ab")
    check_index_result(i1)

def test_index_tolist() -> None:
    i1 = pd.Index([1,2,3])
    l1 = i1.tolist()
    i2 = i1.to_list()
    