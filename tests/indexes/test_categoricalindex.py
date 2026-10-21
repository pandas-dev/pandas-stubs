from __future__ import annotations

from typing import assert_type

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
        assert_type(ci.set_categories(["a", "b", "c", "d"]), pd.CategoricalIndex),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.set_categories(["a", "b", "c", "d"], True), pd.CategoricalIndex),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.set_categories(["x", "y", "z"], None, True), pd.CategoricalIndex
        ),
        pd.CategoricalIndex,
    )
    check(
        assert_type(
            ci.set_categories(["x", "y", "z"], ordered=False, rename=True),
            pd.CategoricalIndex,
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
        assert_type(ci.rename_categories(["x", "y", "z"]), pd.CategoricalIndex),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.rename_categories({"a": "x"}), pd.CategoricalIndex),
        pd.CategoricalIndex,
    )
    check(
        assert_type(ci.rename_categories(str.upper), pd.CategoricalIndex),
        pd.CategoricalIndex,
    )
