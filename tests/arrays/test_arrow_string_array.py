from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.string_arrow import ArrowStringArray
import pyarrow as pa
import pytest
from typing_extensions import assert_type

from pandas._libs.missing import NAType

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import PyArrowStrDtypeArg
from tests.dtypes import PYARROW_STRING_ARGS
from tests.utils import powerset


@pytest.mark.parametrize("data", powerset(["pd", np.str_("pd")]))
@pytest.mark.parametrize(("dtype", "target_dtype"), PYARROW_STRING_ARGS.items())
def test_construction_dtype(
    data: tuple[str | np.str_, ...], dtype: PyArrowStrDtypeArg, target_dtype: type
) -> None:
    dtype_notna = target_dtype if data else None
    check(pd.array([*data], dtype), ArrowStringArray, dtype_notna)
    check(pd.array([*data, *data], dtype), ArrowStringArray, dtype_notna)

    dtype_na = target_dtype if data else NAType
    check(pd.array([*data, np.nan], dtype), ArrowStringArray, dtype_na)
    check(pd.array([*data, *data, np.nan], dtype), ArrowStringArray, dtype_na)

    if TYPE_CHECKING:
        assert_type(pd.array([], pd.StringDtype("pyarrow")), ArrowStringArray)
        assert_type(pd.array([], "string[pyarrow]"), ArrowStringArray)

        assert_type(pd.array([np.nan], pd.StringDtype("pyarrow")), ArrowStringArray)
        assert_type(pd.array([np.nan], "string[pyarrow]"), ArrowStringArray)

        assert_type(pd.array(["1"], pd.StringDtype("pyarrow")), ArrowStringArray)
        assert_type(pd.array(["1"], "string[pyarrow]"), ArrowStringArray)

        assert_type(pd.array(["1", "2"], pd.StringDtype("pyarrow")), ArrowStringArray)
        assert_type(pd.array(["1", "2"], "string[pyarrow]"), ArrowStringArray)

        assert_type(
            pd.array(["1", np.nan], pd.StringDtype("pyarrow")), ArrowStringArray
        )
        assert_type(pd.array(["1", np.nan], "string[pyarrow]"), ArrowStringArray)

        assert_type(
            pd.array([np.str_("1")], pd.StringDtype("pyarrow")), ArrowStringArray
        )
        assert_type(pd.array([np.str_("1")], "string[pyarrow]"), ArrowStringArray)

        assert_type(
            pd.array([np.str_("1"), np.str_("2")], pd.StringDtype("pyarrow")),
            ArrowStringArray,
        )
        assert_type(
            pd.array([np.str_("1"), np.str_("2")], "string[pyarrow]"), ArrowStringArray
        )

        assert_type(
            pd.array([np.str_("1"), np.nan], pd.StringDtype("pyarrow")),
            ArrowStringArray,
        )
        assert_type(
            pd.array([np.str_("1"), np.nan], "string[pyarrow]"), ArrowStringArray
        )

        assert_type(
            pd.array(["1", np.str_("2")], pd.StringDtype("pyarrow")), ArrowStringArray
        )
        assert_type(pd.array([np.str_("1"), "2"], "string[pyarrow]"), ArrowStringArray)


@pytest.mark.parametrize("values", [pa.array(["1"]), pa.chunked_array([["1"]])])
def test_constructor(
    values: "pa.StringArray | pa.ChunkedArray[pa.StringScalar]",
) -> None:
    check(ArrowStringArray(values), ArrowStringArray)

    if TYPE_CHECKING:
        assert_type(ArrowStringArray(pa.array(["1"])), ArrowStringArray)
        assert_type(ArrowStringArray(pa.chunked_array([["1"]])), ArrowStringArray)

    if TYPE_CHECKING_INVALID_USAGE:
        _list = ArrowStringArray([1])  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _tuple = ArrowStringArray((1,))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _np_obj = ArrowStringArray(np.array(["1"], np.object_))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _pa_arr = ArrowStringArray(pa.array([["1"]]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _pd_arr = ArrowStringArray(pd.array(["1"]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _pd_str = ArrowStringArray(pd.array(["1"], "string[pyarrow]"))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _i = ArrowStringArray(pd.Index([1]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _s = ArrowStringArray(pd.Series([1]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_dtype() -> None:
    arr = pd.array(["a"], "string[pyarrow]")
    check(assert_type(arr.dtype, "pd.StringDtype[Literal['pyarrow']]"), pd.StringDtype)
    assert assert_type(arr.dtype.storage, Literal["pyarrow"]) == "pyarrow"
