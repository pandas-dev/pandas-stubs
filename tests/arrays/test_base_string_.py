from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    assert_type,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import BaseStringArray
import pytest

from pandas._libs.missing import NAType

from tests import check
from tests._typing import (
    BuiltinStrDtypeArg,
    PandasBaseStrDtypeArg,
)
from tests.dtypes import (
    PANDAS_BASE_STRING_ARGS,
    PYTHON_STRING_ARGS,
)
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


@pytest.mark.parametrize("data", powerset(["pd", np.str_("pd")]))
@pytest.mark.parametrize(
    ("dtype", "target_dtype"), (PYTHON_STRING_ARGS | PANDAS_BASE_STRING_ARGS).items()
)
def test_construction_dtype(
    data: tuple[str | np.str_, ...],
    dtype: BuiltinStrDtypeArg | PandasBaseStrDtypeArg,
    target_dtype: type,
) -> None:
    is_builtin_str = dtype in PYTHON_STRING_ARGS

    dtype_notna = target_dtype if data else None
    check(pd.array([*data], dtype), BaseStringArray, dtype_notna)
    check(pd.array([*data, *data], dtype), BaseStringArray, dtype_notna)

    dtype_na = (
        target_dtype
        if data
        # pandas-dev/pandas#63567 Pandas 3.0 gives StringDtype(na_value=nan) if dtype is str or "str"
        else float if is_builtin_str else NAType
    )
    check(pd.array([*data, np.nan], dtype), BaseStringArray, dtype_na)
    check(pd.array([*data, *data, np.nan], dtype), BaseStringArray, dtype_na)

    if TYPE_CHECKING:
        assert_type(pd.array([], str), BaseStringArray)
        assert_type(pd.array([], "str"), BaseStringArray)

        assert_type(pd.array([], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array([], "string"), BaseStringArray)

        assert_type(pd.array([np.nan], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array([np.nan], "string"), BaseStringArray)

        assert_type(pd.array(["1"], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array(["1"], "string"), BaseStringArray)

        assert_type(pd.array(["1", "2"], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array(["1", "2"], "string"), BaseStringArray)

        assert_type(pd.array(["1", np.nan], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array(["1", np.nan], "string"), BaseStringArray)

        assert_type(pd.array([np.str_("1")], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array([np.str_("1")], "string"), BaseStringArray)

        assert_type(
            pd.array([np.str_("1"), np.str_("2")], pd.StringDtype()), BaseStringArray
        )
        assert_type(pd.array([np.str_("1"), np.str_("2")], "string"), BaseStringArray)

        assert_type(pd.array([np.str_("1"), np.nan], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array([np.str_("1"), np.nan], "string"), BaseStringArray)

        assert_type(pd.array(["1", np.str_("2")], pd.StringDtype()), BaseStringArray)
        assert_type(pd.array([np.str_("1"), "2"], "string"), BaseStringArray)


def test_dtype() -> None:
    arr_string = pd.array(["a"], "string")
    check(assert_type(arr_string.dtype, pd.StringDtype), pd.StringDtype)
    assert assert_type(arr_string.dtype.storage, Literal["python", "pyarrow"]) in {
        "python",
        "pyarrow",
    }

    arr_str = pd.array([pd.NA], str)
    check(assert_type(arr_str, BaseStringArray), BaseStringArray, float)
    assert pd.isna(assert_type(arr_str.dtype.na_value, NAType | float))
