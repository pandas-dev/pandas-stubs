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
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.string_ import BaseStringArray
import pytest
from typing_extensions import assert_type

from pandas._libs.missing import NAType

from tests import (
    PD_LTE_23,
    check,
)
from tests._typing import PandasBaseStrDtypeArg
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
    # TODO: pandas-dev/pandas#54466 add BuiltinStrDtypeArg after Pandas 3.0
    dtype: PandasBaseStrDtypeArg,
    target_dtype: type,
) -> None:
    is_builtin_str = dtype in PYTHON_STRING_ARGS
    is_numpy_extension_array = PD_LTE_23 and is_builtin_str
    # TODO: pandas-dev/pandas#54466 should give BaseStringArray or even ArrowStringAray after Pandas 3.0
    target_type = NumpyExtensionArray if is_numpy_extension_array else BaseStringArray

    dtype_notna = target_dtype if data else None
    check(pd.array([*data], dtype), target_type, dtype_notna)
    check(pd.array([*data, *data], dtype), target_type, dtype_notna)

    dtype_na = (
        target_dtype
        # TODO: pandas-dev/pandas#54466 drop `or is_numpy_extension_array` after Pandas 3.0
        if data or is_numpy_extension_array
        # TODO: pandas-dev/pandas#63567 Pandas 3.0 gives StringDtype(na_value=nan) for some reason
        else float if is_builtin_str else NAType
    )
    check(pd.array([*data, np.nan], dtype), target_type, dtype_na)
    check(pd.array([*data, *data, np.nan], dtype), target_type, dtype_na)

    if TYPE_CHECKING:
        # TODO: pandas-dev/pandas#54466 should give BaseStringArray or even ArrowStringAray after 3.0
        # The following one still gives NumpyExtensionArray because issubclass(str, object),
        # and pd.array([], object) gives NumpyExtensionArray
        assert_type(pd.array([], str), NumpyExtensionArray)
        pd.array([], "str")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]

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
    arr = pd.array(["a"], "string")
    check(assert_type(arr.dtype, pd.StringDtype), pd.StringDtype)
    assert assert_type(arr.dtype.storage, Literal["python", "pyarrow"]) in {
        "python",
        "pyarrow",
    }
