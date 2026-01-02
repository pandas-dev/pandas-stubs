from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringArray
import pytest
from typing_extensions import assert_type

from pandas._libs.missing import NAType

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    PandasStrDtypeArg,
    np_ndarray_object,
)
from tests.dtypes import PANDAS_STRING_ARGS
from tests.utils import powerset


@pytest.mark.parametrize("data", powerset(["pd", np.str_("pd")]))
@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_STRING_ARGS.items())
def test_construction_dtype(
    data: tuple[str | np.str_, ...], dtype: PandasStrDtypeArg, target_dtype: type
) -> None:
    dtype_notna = target_dtype if data else None
    check(pd.array([*data], dtype), StringArray, dtype_notna)
    check(pd.array([*data, *data], dtype), StringArray, dtype_notna)

    dtype_na = target_dtype if data else NAType
    check(pd.array([*data, np.nan], dtype), StringArray, dtype_na)
    check(pd.array([*data, *data, np.nan], dtype), StringArray, dtype_na)

    if TYPE_CHECKING:
        assert_type(pd.array([], pd.StringDtype("python")), StringArray)
        assert_type(pd.array([], "string[python]"), StringArray)

        assert_type(pd.array([np.nan], pd.StringDtype("python")), StringArray)
        assert_type(pd.array([np.nan], "string[python]"), StringArray)

        assert_type(pd.array(["1"], pd.StringDtype("python")), StringArray)
        assert_type(pd.array(["1"], "string[python]"), StringArray)

        assert_type(pd.array(["1", "2"], pd.StringDtype("python")), StringArray)
        assert_type(pd.array(["1", "2"], "string[python]"), StringArray)

        assert_type(pd.array(["1", np.nan], pd.StringDtype("python")), StringArray)
        assert_type(pd.array(["1", np.nan], "string[python]"), StringArray)

        assert_type(pd.array([np.str_("1")], pd.StringDtype("python")), StringArray)
        assert_type(pd.array([np.str_("1")], "string[python]"), StringArray)

        assert_type(
            pd.array([np.str_("1"), np.str_("2")], pd.StringDtype("python")),
            StringArray,
        )
        assert_type(
            pd.array([np.str_("1"), np.str_("2")], "string[python]"), StringArray
        )

        assert_type(
            pd.array([np.str_("1"), np.nan], pd.StringDtype("python")), StringArray
        )
        assert_type(pd.array([np.str_("1"), np.nan], "string[python]"), StringArray)

        assert_type(
            pd.array(["1", np.str_("2")], pd.StringDtype("python")), StringArray
        )
        assert_type(pd.array([np.str_("1"), "2"], "string[python]"), StringArray)


@pytest.mark.parametrize(
    "values", [np.array(["1"], np.object_), pd.array(["1"], "string[python]")]
)
def test_constructor(values: np_ndarray_object | StringArray) -> None:
    check(StringArray(values), StringArray)

    if TYPE_CHECKING:
        assert_type(StringArray(np.array(["1"], np.object_)), StringArray)
        assert_type(StringArray(pd.array(["1"], "string[python]")), StringArray)

    if TYPE_CHECKING_INVALID_USAGE:
        _list = StringArray([1])  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _tuple = StringArray((1,))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        np_str = np.array(["1"], np.str_)
        _np_str = StringArray(np_str)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _pd_arr = StringArray(pd.array(["1"]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _i = StringArray(pd.Index([1]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _s = StringArray(pd.Series([1]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_dtype() -> None:
    arr = pd.array(["a"], "string[python]")
    check(assert_type(arr.dtype, "pd.StringDtype[Literal['python']]"), pd.StringDtype)
    assert assert_type(arr.dtype.storage, Literal["python"]) == "python"
