"""Test module for methods in pandas.core.arrays.categorical."""

from typing import (
    assert_type,
)

import numpy as np
import pandas as pd
from pandas import Categorical
from pandas.api.typing.aliases import (
    Ordered,
)
from pandas.core.arrays.categorical import CategoricalDtype
from pandas.core.indexes.base import Index

from pandas._libs.missing import NAType

from tests import check
from tests._typing import (
    np_1darray,
    np_1darray_bool,
)


def test_construction_array_like() -> None:
    # TODO: https://github.com/facebook/pyrefly/issues/3891
    check(
        assert_type(  # pyrefly: ignore[assert-type]
            pd.array(pd.Categorical([1])),
            "Categorical[int]",
        ),
        Categorical,
    )
    check(assert_type(pd.array(pd.CategoricalIndex([1])), Categorical), Categorical)


def test_construction_dtype() -> None:
    check(assert_type(pd.array([], pd.CategoricalDtype()), Categorical), Categorical)
    check(
        assert_type(pd.array(np.array([1]), pd.CategoricalDtype()), Categorical),
        Categorical,
    )
    check(
        assert_type(pd.array(pd.array([1]), pd.CategoricalDtype()), Categorical),
        Categorical,
    )
    check(
        assert_type(pd.array(pd.Index([1]), pd.CategoricalDtype()), Categorical),
        Categorical,
    )
    check(
        assert_type(pd.array(pd.Series([1]), pd.CategoricalDtype()), Categorical),
        Categorical,
    )


def test_constructor() -> None:
    """Test init method for Categorical."""
    dd = ["a", "b", "c", "a"]
    cat = Categorical(dd)
    check(assert_type(cat, "Categorical[str]"), Categorical, str)

    values = np.array(["a", "b", "c", "a"])
    cat_np = Categorical(values)
    # np.array() is typed as ndarray[Any, Any] by numpy stubs, so mypy cannot infer
    # the element type; the actual type is Categorical[str]
    # TODO: https://github.com/facebook/pyrefly/issues/3891
    check(assert_type(cat_np, "Categorical[str]"), Categorical)  # type: ignore[assert-type] # pyrefly: ignore[assert-type]

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
    check(assert_type(cat, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=np.array(["a", "b", "c", "d"]))
    # TODO: https://github.com/facebook/pyrefly/issues/3891
    check(
        assert_type(cat, "Categorical[str]"),  # pyrefly: ignore[assert-type]
        Categorical,
    )

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)
    check(assert_type(cat, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=False)
    check(assert_type(cat, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    check(assert_type(cat, "Categorical[str]"), Categorical)

    cat = Categorical(
        values=["x", "y", "z", "x"],
        categories=["x", "y", "z"],
        ordered=True,
        copy=True,
    )
    check(assert_type(cat, "Categorical[str]"), Categorical)

    dtype = pd.CategoricalDtype(categories=["x", "y", "z"], ordered=True)
    cat = Categorical(
        values=["x", "y", "z", "x"],
        dtype=dtype,
        copy=True,
    )
    check(assert_type(cat, "Categorical[str]"), Categorical)

    cat_int = Categorical([1, 2, 3, 1, 2])
    # TODO: https://github.com/facebook/pyrefly/issues/3891
    check(
        assert_type(cat_int, "Categorical[int]"),  # pyrefly: ignore[assert-type]
        Categorical,
    )

    cat_mixed = Categorical(["a", 1, "b", 2])
    # https://github.com/facebook/pyrefly/issues/3891
    check(
        assert_type(cat_mixed, Categorical), Categorical  # pyrefly: ignore[assert-type]
    )

    cat_empty = Categorical([])
    check(assert_type(cat_empty, Categorical), Categorical)

    cat = Categorical(["a", "b", "c"], categories=None)
    check(assert_type(cat, "Categorical[str]"), Categorical)

    cat1 = Categorical(["a", "b", "c"])
    cat = Categorical(cat1)
    check(assert_type(cat, "Categorical[str]"), Categorical)

    values_series = pd.Series(["a", "b", "c", "a"])
    cat = Categorical(values_series)
    check(assert_type(cat, "Categorical[str]"), Categorical)

    values_index = pd.Index(["a", "b", "c", "a"])
    cat = Categorical(values_index)
    check(assert_type(cat, "Categorical[str]"), Categorical)


def test_categorical_dtype() -> None:
    """Test dtype property for Categorical."""
    values_index = pd.Index(["a", "b", "c", "a"])
    cat = Categorical(values_index)
    check(assert_type(cat.dtype, "CategoricalDtype[str]"), CategoricalDtype)


def test_categorical_properties() -> None:
    """Test properties for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])

    check(assert_type(cat.categories, Index), Index)
    check(assert_type(cat.ordered, Ordered), bool)
    check(assert_type(cat.dtype, "CategoricalDtype[str]"), CategoricalDtype)
    check(assert_type(cat.nbytes, int), int)
    check(assert_type(cat.codes, np_1darray[np.signedinteger]), np_1darray, np.integer)


def test_categorical_tolist() -> None:
    """Test tolist method for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.tolist(), "list[str]"), list)


def test_categorical_from_codes() -> None:
    """Test from_codes class method for Categorical."""
    codes = [0, 1, 2, 0, 1]
    categories = Index(["a", "b", "c"])
    cat = Categorical.from_codes(codes, categories)
    check(assert_type(cat, "Categorical[str]"), Categorical)

    cat1 = Categorical.from_codes(codes, Index([0, 1, 2]))
    check(assert_type(cat1, "Categorical[int]"), Categorical)

    dtype = CategoricalDtype(categories=["x", "y", "z"], ordered=True)
    cat2 = Categorical.from_codes(codes, dtype=dtype)
    check(assert_type(cat2, Categorical), Categorical)


def test_categorical_from_codes_ndarray() -> None:
    """Test from_codes class method for Categorical."""
    codes = np.array([0, 1, 2, 0, 1])
    categories = Index(["a", "b", "c"])
    cat = Categorical.from_codes(codes, categories)
    check(assert_type(cat, "Categorical[str]"), Categorical)


def test_categorical_from_codes_series() -> None:
    """Test from_codes class method for Categorical."""
    codes = pd.Series([0, 1, 2, 0, 1])
    categories = Index(["a", "b", "c"])
    cat = Categorical.from_codes(codes, categories)
    check(assert_type(cat, "Categorical[str]"), Categorical)


def test_categorical_set_ordered() -> None:
    """Test set_ordered for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.set_ordered(True), "Categorical[str]"), Categorical)


def test_categorical_as_ordered() -> None:
    """Test as_ordered for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.as_ordered(), "Categorical[str]"), Categorical)


def test_categorical_as_unordered() -> None:
    """Test as_unordered for Categorical."""
    cat = Categorical(["a", "b", "c", "a"])
    check(assert_type(cat.as_unordered(), "Categorical[str]"), Categorical)


def test_categorical_set_categories() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(["a", "b", "c", "d"])
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(["x", "y", "z"])
    check(assert_type(result, "Categorical[str]"), Categorical)


def test_categorical_set_categories_ndarray() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    new_cats = np.array(["a", "b", "c", "d", "e"])
    result = cat.set_categories(new_cats)
    check(assert_type(result, "Categorical[str]"), Categorical)


def test_categorical_set_categories_index() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    new_cats = Index(["a", "b", "c", "d"])
    result = cat.set_categories(new_cats)
    check(assert_type(result, "Categorical[str]"), Categorical)


def test_categorical_set_categories_series() -> None:
    """Test set_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    new_cats = pd.Series(["a", "b", "c", "d"])
    result = cat.set_categories(new_cats)
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(("a", "b", "c", "d"))
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"])
    result = cat.set_categories(["a", "b"])
    check(assert_type(result, "Categorical[str]"), Categorical)


def test_categorical_rename_categories() -> None:
    """Test rename_categories for Categorical."""
    cat = Categorical(["a", "b", "c"])
    result = cat.rename_categories({"a": "d"})
    check(assert_type(result, "Categorical[str]"), Categorical)


def test_categorical_reorder_categorical() -> None:
    """Test reorder_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.reorder_categories(["b", "a", "c"])
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.reorder_categories(pd.Series(["b", "a", "c"]))
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.reorder_categories(pd.Index(["b", "a", "c"]))
    check(assert_type(result, "Categorical[str]"), Categorical)

    # np.array is not subtyped statically so will return Categorical
    cat1 = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result1 = cat1.reorder_categories(np.array(["b", "a", "c"]))
    check(assert_type(result1, Categorical), Categorical)


def test_categorical_add_categories() -> None:
    """Test add_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(pd.Index(["d"]))
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(np.array([1]))
    # TODO: pandas-dev/pandas-stubs#1415 Fix this
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(pd.Series(["d"]))
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.add_categories(pd.Index(["d"]))
    check(assert_type(result, "Categorical[str]"), Categorical)


def test_categorical_remove_categories() -> None:
    """Test remove_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(pd.Index(["a"]))
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(np.array(["a"]))
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(pd.Series(["b"]))
    check(assert_type(result, "Categorical[str]"), Categorical)

    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_categories(pd.Index(["a"]))
    check(assert_type(result, "Categorical[str]"), Categorical)


def test_categorical_remove_unused_categories() -> None:
    """Test remove_unused_categories for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=None)
    result = cat.remove_unused_categories()
    check(assert_type(result, "Categorical[str]"), Categorical)


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
    check(assert_type(cat.sort_values(), "Categorical[str]"), Categorical)
    check(assert_type(cat.sort_values(inplace=False), "Categorical[str]"), Categorical)
    check(assert_type(cat.sort_values(inplace=True), None), type(None))

    check(assert_type(cat.sort_values(ascending=True), "Categorical[str]"), Categorical)
    check(
        assert_type(cat.sort_values(ascending=False), "Categorical[str]"), Categorical
    )

    check(
        assert_type(
            cat.sort_values(ascending=True, na_position="first"), "Categorical[str]"
        ),
        Categorical,
    )
    check(
        assert_type(cat.sort_values(na_position="last"), "Categorical[str]"),
        Categorical,
    )


def test_categorical_min_max() -> None:
    """Test min/max for Categorical."""
    cat = Categorical(["a", "b", "c"], categories=["a", "b", "c"], ordered=True)

    check(assert_type(cat.min(), str | NAType), str)
    check(assert_type(cat.max(), str | NAType), str)

    cat_w_nan = Categorical([], ordered=True)

    # below the returns is nan, not pd.NA
    check(assert_type(cat_w_nan.min(), object | NAType), float)
    check(assert_type(cat_w_nan.max(), object | NAType), float)


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
