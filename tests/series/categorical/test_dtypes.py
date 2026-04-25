from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    assert_type,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.categorical import Categorical
import pytest

from tests import check
from tests._typing import CategoryDtypeArg
from tests.dtypes import ASTYPE_CATEGORICAL_ARGS

if TYPE_CHECKING:
    from pandas._typing import CategoricalSeries


@pytest.mark.parametrize(
    "cast_arg, target_type", ASTYPE_CATEGORICAL_ARGS.items(), ids=repr
)
def test_astype_categorical(cast_arg: CategoryDtypeArg, target_type: type) -> None:
    s = pd.Series(["a", "b"])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # pandas category
        assert_type(s.astype(pd.CategoricalDtype()), "pd.Series[pd.CategoricalDtype]")
        assert_type(s.astype(cast_arg), "pd.Series[pd.CategoricalDtype]")


def test_categorical_series_alias() -> None:
    # GH1415: ``CategoricalSeries`` is the public alias for
    # ``Series[CategoricalDtype]``. It must be assignment-compatible with the
    # parameterized form so existing user code keeps type-checking.
    s_cat = pd.Series([1, 2, 3], dtype="category")
    check(assert_type(s_cat, "pd.Series[pd.CategoricalDtype]"), pd.Series, np.integer)

    if TYPE_CHECKING:
        s_alias: CategoricalSeries = s_cat
        s_param: pd.Series[pd.CategoricalDtype] = s_cat
        # Reverse direction: a value typed as the alias is assignable to the
        # parameterized form.
        s_back: pd.Series[pd.CategoricalDtype] = s_alias
        # Suppress unused-variable warnings without runtime evaluation.
        del s_alias, s_param, s_back


def test_categorical_cat_accessor() -> None:
    # GH1415: exercise common ``.cat`` accessor methods on a categorical
    # ``Series`` and confirm their return types are well-defined.
    # ``Sequence[str]`` triggers the str-data overload first, so build the
    # series via ``astype`` to keep the categorical parameterization.
    s_cat = pd.Series(["a", "b", "a", "c"]).astype("category")
    check(assert_type(s_cat, "pd.Series[pd.CategoricalDtype]"), pd.Series)

    # ``codes`` returns a ``Series[int]``; ``categories`` returns an Index.
    check(assert_type(s_cat.cat.codes, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s_cat.cat.categories, pd.Index), pd.Index)
    check(assert_type(s_cat.cat.ordered, "bool | None"), bool)

    # ``Series.array`` for a categorical Series narrows to ``Categorical``.
    check(assert_type(s_cat.array, Categorical), Categorical)


def test_categorical_series_copy() -> None:
    # GH1415: ``Series.copy`` on a ``CategoricalSeries`` must preserve the
    # categorical parameterization so further operations stay typed.
    s_cat = pd.Series([1, 2, 3], dtype="category")
    s_copy = s_cat.copy()
    check(assert_type(s_copy, "pd.Series[pd.CategoricalDtype]"), pd.Series)
