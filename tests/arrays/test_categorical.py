"""Test module for methods in pandas.core.arrays.categorical."""

import numpy as np
import pandas as pd
from pandas import Categorical
from pandas.core.arrays.categorical import CategoricalDtype
from pandas.core.indexes.base import Index
from typing_extensions import assert_type

from pandas._libs.missing import NAType
from pandas._typing import Ordered
from pandas._typing import Scalar  # noqa: F401

from tests import (
    check,
    pytest_warns_bounded,
)
from tests._typing import (
    np_1darray,
    np_1darray_bool,
)


def test_constructor() -> None:
    """Test init method for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat, Categorical), Categorical)

    values = np.array(["a", "b", "c", "a"])
    cat = Categorical(values)
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=np.array(["a", "b", "c", "d"]))
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=False)
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(
        values=["x", "y", "z", "x"],
        categories=["x", "y", "z"],
        ordered=True,
        copy=True,
    )
    check(assert_type(cat, Categorical), Categorical)

    dtype = CategoricalDtype(categories=["x", "y", "z"], ordered=True)
    cat = Categorical(
        values=["x", "y", "z", "x"],
        dtype=dtype,
        copy=True,
    )
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical([1, 2, 3, 1, 2])
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(["a", 1, "b", 2])
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical([])
    check(assert_type(cat, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=None)
    check(assert_type(cat, Categorical), Categorical)

    cat1 = Categorical(["a", "b", "c"])
    cat = Categorical(cat1)
    check(assert_type(cat, Categorical), Categorical)

    values_series = pd.Series(["a", "b", "c", "a"])
    cat = Categorical(values_series)
    check(assert_type(cat, Categorical), Categorical)

    values_index = pd.Index(["a", "b", "c", "a"])
    cat = Categorical(values_index)
    check(assert_type(cat, Categorical), Categorical)


def test_categorical_properties() -> None:
    """Test properties for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])

    check(assert_type(cat.categories, Index), Index)
    check(assert_type(cat.ordered, Ordered), bool)
    check(assert_type(cat.dtype, CategoricalDtype), CategoricalDtype)
    check(assert_type(cat.nbytes, int), int)
    check(assert_type(cat.codes, np_1darray[np.signedinteger]), np_1darray, np.integer)


def test_categorical_tolist() -> None:
    """Test tolist method for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.tolist(), "list[Scalar]"), list)

    with pytest_warns_bounded(
        FutureWarning,
        r"Categorical.to_list is deprecated and will.*",
        lower="2.3.0",
        upper="2.99",
    ):
        check(assert_type(cat.to_list(), "list[Scalar]"), list)


def test_categorical_from_codes() -> None:
    """Test from_codes class method for Categorical."""
    codes = [0, 1, 2, 0, 1]
    categories = Index(["a", "b", "c"])
    cat = Categorical.from_codes(codes, categories)
    check(assert_type(cat, Categorical), Categorical)


def test_categorical_from_codes_ndarray() -> None:
    """Test from_codes class method for Categorical."""
    codes = np.array([0, 1, 2, 0, 1])
    categories = Index(["a", "b", "c"])
    cat = Categorical.from_codes(codes, categories)
    check(assert_type(cat, Categorical), Categorical)


def test_categorical_from_codes_series() -> None:
    """Test from_codes class method for Categorical."""
    codes = pd.Series([0, 1, 2, 0, 1])
    categories = Index(["a", "b", "c"])
    cat = Categorical.from_codes(codes, categories)
    check(assert_type(cat, Categorical), Categorical)


def test_categorical_set_ordered() -> None:
    """Test set_ordered for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.set_ordered(True), Categorical), Categorical)


def test_categorical_as_ordered() -> None:
    """Test as_ordered for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.as_ordered(), Categorical), Categorical)


def test_categorical_as_unordered() -> None:
    """Test as_unordered for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.as_unordered(), Categorical), Categorical)


def test_categorical_set_categories() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(["a", "b", "c", "d"])
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(["x", "y", "z"])
    check(assert_type(result, Categorical), Categorical)


def test_categorical_set_categories_ndarray() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    new_cats = np.array(["a", "b", "c", "d", "e"])
    result = cat.set_categories(new_cats)
    check(assert_type(result, Categorical), Categorical)


def test_categorical_set_categories_index() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    new_cats = Index(["a", "b", "c", "d"])
    result = cat.set_categories(new_cats)
    check(assert_type(result, Categorical), Categorical)


def test_categorical_set_categories_series() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    new_cats = pd.Series(["a", "b", "c", "d"])
    result = cat.set_categories(new_cats)
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(("a", "b", "c", "d"))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(["a", "b"])
    check(assert_type(result, Categorical), Categorical)


def test_categorical_rename_categories() -> None:
    """Test rename_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    result = cat.rename_categories({"a": "d"})
    check(assert_type(result, Categorical), Categorical)


def test_categorical_reorder_categorical() -> None:
    """Test reorder_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.reorder_categories(["b", "a", "c"])
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.reorder_categories(pd.Series(["b", "a", "c"]))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.reorder_categories(pd.Index(["b", "a", "c"]))
    check(assert_type(result, Categorical), Categorical)


def test_categorical_add_categories() -> None:
    """Test add_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(pd.Index(["d"]))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(np.array([1]))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(pd.Series(["d"]))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(pd.Index(["d"]))
    check(assert_type(result, Categorical), Categorical)


def test_categorical_remove_categories() -> None:
    """Test remove_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(pd.Index(["a"]))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(np.array(["a"]))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(pd.Series(["b"]))
    check(assert_type(result, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(pd.Index(["a"]))
    check(assert_type(result, Categorical), Categorical)


def test_categorical_remove_unused_categories() -> None:
    """Test remove_unused_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_unused_categories()
    check(assert_type(result, Categorical), Categorical)


def test_categorical_memory_usage() -> None:
    """Test memory_usage for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    check(assert_type(cat.memory_usage(), int), int)


def test_categorical_isna_isnull() -> None:
    """Test isna/isnull/notna/notnull for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)

    check(assert_type(cat.isna(), np_1darray_bool), np_1darray_bool)
    check(assert_type(cat.isnull(), np_1darray_bool), np_1darray_bool)
    check(assert_type(cat.notna(), np_1darray_bool), np_1darray_bool)
    check(assert_type(cat.notnull(), np_1darray_bool), np_1darray_bool)


def test_categorical_sort_values() -> None:
    """Test sort_values for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    check(assert_type(cat.sort_values(), Categorical), Categorical)
    check(assert_type(cat.sort_values(inplace=False), Categorical), Categorical)
    check(assert_type(cat.sort_values(inplace=True), None), type(None))

    check(assert_type(cat.sort_values(ascending=True), Categorical), Categorical)
    check(assert_type(cat.sort_values(ascending=False), Categorical), Categorical)

    check(
        assert_type(cat.sort_values(ascending=True, na_position="first"), Categorical),
        Categorical,
    )
    check(assert_type(cat.sort_values(na_position="last"), Categorical), Categorical)


def test_categorical_min_max() -> None:
    """Test min/max for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)

    check(assert_type(cat.min(), Scalar | NAType), str)
    check(assert_type(cat.max(), Scalar | NAType), str)

    cat_w_nan = Categorical([], ordered=True)

    # below the returns is nan, not pd.NA
    check(assert_type(cat_w_nan.min(), Scalar | NAType), float)
    check(assert_type(cat_w_nan.max(), Scalar | NAType), float)


def test_categorical_equals() -> None:
    """Test equals for Categorical."""
    cat_l = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)
    cat_r = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)

    check(assert_type(cat_l.equals(cat_r), bool), bool)


def test_categorical_describe() -> None:
    """Test describe for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)

    check(assert_type(cat.describe(), pd.DataFrame), pd.DataFrame)


def test_categorical_isin() -> None:
    """Test isin for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)

    check(assert_type(cat.isin(["b", 1]), np_1darray_bool), np_1darray_bool)
