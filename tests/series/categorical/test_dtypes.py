from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import CategoryDtypeArg
from tests.dtypes import ASTYPE_CATEGORICAL_ARGS


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
