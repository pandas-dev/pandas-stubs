from __future__ import annotations

import numpy as np
import pandas as pd
from typing_extensions import (
    assert_type,
)

from tests import (
    check,
    np_1darray,
)


def test_categoricalindex_unique() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(
        assert_type(ci.unique(), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )


def test_categoricalindex_reindex() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(
        assert_type(ci.reindex([0, 1]), tuple[pd.Index, np_1darray[np.intp] | None]),
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
