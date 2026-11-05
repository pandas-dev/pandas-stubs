from __future__ import annotations

from collections import UserDict
from typing import Any, assert_type

import pandas as pd

from tests import check
from tests._typing import np_1darray_intp


def test_categoricalindex_unique() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(
        assert_type(ci.unique(), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )


def test_categoricalindex_reindex() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(
        assert_type(ci.reindex([0, 1]), tuple[pd.Index, np_1darray_intp | None]),
        tuple,
    )


def test_categoricalindex_delete() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(assert_type(ci.delete(0), "pd.CategoricalIndex[str]"), pd.CategoricalIndex)
    check(
        assert_type(ci.delete([0, 1]), "pd.CategoricalIndex[str]"), pd.CategoricalIndex
    )


def test_categoricalindex_insert() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(assert_type(ci.insert(0, "c"), pd.Index), pd.Index)


def test_categoricalindex_ordered() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(assert_type(ci.ordered, bool | None), bool)
    check(
        assert_type(pd.CategoricalIndex(["a"], ordered=True).ordered, bool | None), bool
    )
    check(assert_type(ci.as_ordered(), "pd.CategoricalIndex[str]"), pd.CategoricalIndex)
    check(
        assert_type(ci.as_unordered(), "pd.CategoricalIndex[str]"), pd.CategoricalIndex
    )


def test_categoricalindex_categories_methods() -> None:
    ci = pd.CategoricalIndex(["a", "b", "c"])
    check(
        assert_type(ci.add_categories(["d"]), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.remove_categories("a"), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.remove_categories(["a"]), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.remove_unused_categories(), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.reorder_categories(["c", "b", "a"], ordered=True),
            "pd.CategoricalIndex[str]",
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.set_categories(["a", "b", "c", "d"]), "pd.CategoricalIndex[str]"
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.set_categories(["a", "b", "c", "d"], True), "pd.CategoricalIndex[str]"
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.set_categories(["x", "y", "z"], None, True),
            "pd.CategoricalIndex[str]",
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.set_categories(["x", "y", "z"], ordered=False, rename=True),
            "pd.CategoricalIndex[str]",
        ),
        pd.CategoricalIndex,
    )
    ci_str = pd.CategoricalIndex(["a", "b", "c"])
    check(
        assert_type(
            ci_str.set_categories([1, 2, 3], rename=True),
            "pd.CategoricalIndex[int]",
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.reorder_categories(["c", "b", "a"], True), "pd.CategoricalIndex[str]"
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.add_categories("d"), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.rename_categories(["x", "y", "z"]), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.rename_categories({"a": "x"}), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.rename_categories(str.upper), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    ci_dates = pd.CategoricalIndex(
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    )
    check(
        assert_type(
            ci_dates.rename_categories(
                [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")]
            ),
            "pd.CategoricalIndex[pd.Timestamp]",
        ),
        pd.CategoricalIndex,
    )
    mixed: list[int | str] = [1, "a", 2]
    check(
        assert_type(ci.rename_categories(mixed), "pd.CategoricalIndex[int | str]"),
        pd.CategoricalIndex,
    )
    ci_int = pd.CategoricalIndex([1, 2, 3])
    ts = [
        pd.Timestamp("2020"),
        pd.Timestamp("2021"),
        pd.Timestamp("2022"),
    ]
    ci_int.rename_categories(ts)
    ci_int.rename_categories([1, "a", 2])
    ci_any = pd.Index([1, 2, 3], dtype="category")
    check(
        assert_type(ci_any, "pd.CategoricalIndex[Any]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci_any.rename_categories(ts),
            "pd.CategoricalIndex[Any]",
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci_int.rename_categories(["one", "two", "three"]),
            "pd.CategoricalIndex[str]",
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci_int.rename_categories({1: "one", 2: "two", 3: "three"}),
            "pd.CategoricalIndex[str]",
        ),
        pd.CategoricalIndex,
    )
    mapping = UserDict[int, str]({1: "one", 2: "two", 3: "three"})
    check(
        assert_type(ci_int.rename_categories(mapping), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci_int.rename_categories(str), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )
