from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import BaseStringArray
import pytest
from typing_extensions import assert_type

from tests import check
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("data", powerset(["🐼", np.str_("🐼")], 1))
@pytest.mark.parametrize("missing_values", powerset([np.nan, None, pd.NA]))
def test_construction_sequence(
    data: tuple[str | np.str_, ...],
    missing_values: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
) -> None:
    check(pd.array(typ([*data, *missing_values])), BaseStringArray)

    if TYPE_CHECKING:
        assert_type(pd.array(["🐼", np.str_("🐼")]), BaseStringArray)

        assert_type(pd.array(["🐼", np.str_("🐼"), None]), BaseStringArray)
        assert_type(pd.array(["🐼", np.str_("🐼"), pd.NA]), BaseStringArray)

        assert_type(pd.array(["🐼", np.str_("🐼"), None, pd.NA]), BaseStringArray)

        assert_type(pd.array(("🐼", np.str_("🐼"))), BaseStringArray)
        assert_type(pd.array(("🐼", np.str_("🐼"), pd.NA)), BaseStringArray)

        assert_type(pd.array(UserList(["🐼", np.str_("🐼")])), BaseStringArray)


def test_construction_array_like() -> None:
    np_arr = np.array(["🐼", np.str_("🐼")], np.str_)
    check(assert_type(pd.array(np_arr), BaseStringArray), BaseStringArray)

    check(
        assert_type(pd.array(pd.array(["🐼", np.str_("🐼")])), BaseStringArray),
        BaseStringArray,
    )


def test_construction_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "string"), BaseStringArray), BaseStringArray)
