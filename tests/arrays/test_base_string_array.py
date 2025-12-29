from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import BaseStringArray
import pytest
from typing_extensions import assert_type

from tests import check
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("data", powerset(["pd", np.str_("pd")], 1))
@pytest.mark.parametrize("missing_values", powerset([np.nan, None, pd.NA]))
def test_construction_sequence(
    data: tuple[str | np.str_, ...],
    missing_values: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
) -> None:
    check(pd.array(typ([*data, *missing_values])), BaseStringArray)
    check(pd.array(typ([*data, *data, *missing_values])), BaseStringArray)

    if TYPE_CHECKING:
        assert_type(pd.array(["pd", np.str_("pd")]), BaseStringArray)
        assert_type(pd.array(["pa", "pd"]), BaseStringArray)
        assert_type(pd.array([np.str_("pa"), np.str_("pd")]), BaseStringArray)

        assert_type(pd.array(["pd", np.str_("pd"), None]), BaseStringArray)
        assert_type(pd.array(["pd", np.str_("pd"), pd.NA]), BaseStringArray)
        assert_type(pd.array([np.str_("pa"), np.str_("pd"), pd.NA]), BaseStringArray)

        assert_type(pd.array(["pd", np.str_("pd"), None, pd.NA]), BaseStringArray)
        assert_type(pd.array(["pa", "pd", None, pd.NA]), BaseStringArray)

        assert_type(pd.array(("pd", np.str_("pd"))), BaseStringArray)
        assert_type(pd.array(("pd", np.str_("pd"), pd.NA)), BaseStringArray)

        assert_type(pd.array(UserList(["pd", np.str_("pd")])), BaseStringArray)


def test_construction_array_like() -> None:
    np_arr = np.array(["pd", np.str_("pd")], np.str_)
    check(assert_type(pd.array(np_arr), BaseStringArray), BaseStringArray)

    check(
        assert_type(pd.array(pd.array(["pd", np.str_("pd")])), BaseStringArray),
        BaseStringArray,
    )


def test_construction_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "string"), BaseStringArray), BaseStringArray)


def test_dtype() -> None:
    arr = pd.array(["a"], "string")
    check(assert_type(arr.dtype, pd.StringDtype), pd.StringDtype)
    assert assert_type(arr.dtype.storage, Literal["python", "pyarrow"]) in {
        "python",
        "pyarrow",
    }
