from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    assert_type,
)

import pandas as pd
import pytest

from tests import check
from tests._typing import CategoryDtypeArg
from tests.dtypes import ASTYPE_CATEGORICAL_ARGS

if TYPE_CHECKING:
    from typing import Any  # noqa: F401

    from pandas._stubs_only import (  # pyright: ignore[reportMissingModuleSource]  # isort:skip
        C1,  # noqa: F401
    )


@pytest.mark.parametrize(
    "cast_arg, target_type", ASTYPE_CATEGORICAL_ARGS.items(), ids=repr
)
def test_astype_categorical(cast_arg: CategoryDtypeArg, target_type: type) -> None:
    s = pd.Series(["a", "b"])
    check(s.astype(cast_arg), pd.Series, target_type)

    check(
        assert_type(s.astype(pd.CategoricalDtype()), "pd.Series[C1[Any]]"),
        pd.Series,
        str,
    )
    check(assert_type(s.astype(cast_arg), "pd.Series[C1[Any]]"), pd.Series, str)
