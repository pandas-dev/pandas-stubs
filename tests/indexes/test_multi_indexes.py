from __future__ import annotations

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check
from tests._typing import (
    np_1darray_bool,
    np_1darray_int8,
    np_1darray_intp,
)


def test_multiindex_unique() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    check(assert_type(mi.unique(), pd.MultiIndex), pd.MultiIndex)
    check(assert_type(mi.unique(level=0), pd.Index), pd.Index)


def test_multiindex_set_levels() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    res = mi.set_levels([[10, 20, 30], [40, 50, 60]])
    check(assert_type(res, pd.MultiIndex), pd.MultiIndex)
    res = mi.set_levels([10, 20, 30], level=0)
    check(assert_type(res, pd.MultiIndex), pd.MultiIndex)


def test_multiindex_codes() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    check(assert_type(mi.codes, list[np_1darray_int8]), list)


def test_multiindex_set_codes() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    res = mi.set_codes([[0, 1, 2], [0, 1, 2]])
    check(assert_type(res, pd.MultiIndex), pd.MultiIndex)
    res = mi.set_codes([0, 1, 2], level=0)
    check(assert_type(res, pd.MultiIndex), pd.MultiIndex)


def test_multiindex_view() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    check(assert_type(mi.view(), pd.MultiIndex), pd.MultiIndex)
    check(assert_type(mi.view(np.ndarray), pd.MultiIndex), pd.MultiIndex)


def test_multiindex_remove_unused_levels() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3, 1], [4, 5, 6, 4]])
    res = mi.remove_unused_levels()
    check(assert_type(res, pd.MultiIndex), pd.MultiIndex)


def test_multiindex_levshape() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3, 1], [4, 5, 6, 4]])
    ls = mi.levshape
    check(assert_type(ls, tuple[int, ...]), tuple, int)


def test_multiindex_append() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    check(assert_type(mi.append([mi]), pd.MultiIndex), pd.MultiIndex)
    check(assert_type(mi.append([pd.Index([1, 2])]), pd.Index), pd.Index)


def test_multiindex_drop() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    dropped = mi.drop([1])
    check(assert_type(dropped, pd.MultiIndex), pd.MultiIndex)


def test_multiindex_reorder_levels() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
    reordered = mi.reorder_levels([1, 0])
    check(assert_type(reordered, pd.MultiIndex), pd.MultiIndex)


def test_multiindex_get_locs() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3, 1], [4, 5, 6, 4]])
    locs = mi.get_locs([1, 4])
    check(assert_type(locs, np_1darray_intp), np_1darray_intp)


def test_multiindex_get_loc_level() -> None:
    mi = pd.MultiIndex.from_arrays([[1, 2, 3, 1], [4, 5, 6, 4]])
    res_0, res_1 = mi.get_loc_level(1, level=0)
    check(assert_type(res_0, int | slice | np_1darray_bool), np_1darray_bool)
    check(assert_type(res_1, pd.Index), pd.Index)
