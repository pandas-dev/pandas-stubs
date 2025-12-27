from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    PD_LTE_23,
    check,
)

if TYPE_CHECKING:
    from pandas._libs.missing import NAType
    from pandas._libs.tslibs.nattype import NaTType


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize(
    "data",
    [
        [None],
        [pd.NA],
        [pd.NaT],
        [np.nan, pd.NaT],
        [None, pd.NA],
        [None, pd.NaT],
        [pd.NA, pd.NaT],
        [np.nan, None, pd.NaT],
        [np.nan, pd.NA, pd.NaT],
        [None, pd.NA, pd.NaT],
        [np.nan, None, pd.NA, pd.NaT],
    ],
)
def test_constructor_sequence(
    data: Sequence[Any], typ: Callable[[Sequence[Any]], Sequence[Any]]
) -> None:
    # `pd.NaT in [pd.NA, pd.NaT]` leads to an exception
    if data[-1] is pd.NaT and PD_LTE_23:
        check(pd.array(typ(data)), DatetimeArray)
    else:
        check(pd.array(typ(data)), NumpyExtensionArray)

    if TYPE_CHECKING:
        assert_type(pd.array([None]), NumpyExtensionArray)
        assert_type(pd.array([pd.NA]), NumpyExtensionArray)
        assert_type(pd.array([pd.NaT]), NumpyExtensionArray)

        _10 = [np.nan, pd.NaT]
        assert_type(pd.array(_10), NumpyExtensionArray)
        assert_type(pd.array([None, pd.NA]), NumpyExtensionArray)
        assert_type(pd.array([None, pd.NaT]), NumpyExtensionArray)
        assert_type(pd.array([pd.NA, pd.NaT]), NumpyExtensionArray)

        _20 = [np.nan, None, pd.NaT]
        assert_type(pd.array(_20), NumpyExtensionArray)
        _21 = [np.nan, pd.NA, pd.NaT]
        assert_type(pd.array(_21), NumpyExtensionArray)
        assert_type(pd.array([None, pd.NA, pd.NaT]), NumpyExtensionArray)

        _30 = [np.nan, None, pd.NA, pd.NaT]
        assert_type(pd.array(_30), NumpyExtensionArray)

        assert_type(pd.array(tuple(_30)), NumpyExtensionArray)
        _41 = cast(  # pyright: ignore[reportUnnecessaryCast]
            "UserList[float | NAType | NaTType | None]", UserList(_30)
        )
        assert_type(pd.array(_41), NumpyExtensionArray)


def test_constructor_array_like() -> None:
    data = [1, "🐼"]
    np_arr = np.array(data, np.object_)

    check(assert_type(pd.array(data), NumpyExtensionArray), NumpyExtensionArray)

    check(assert_type(pd.array(np_arr), NumpyExtensionArray), NumpyExtensionArray)

    check(
        assert_type(pd.array(pd.array(data)), NumpyExtensionArray), NumpyExtensionArray
    )

    check(
        assert_type(pd.array(pd.RangeIndex(0, 1)), NumpyExtensionArray),
        NumpyExtensionArray,
    )


def test_constructor_dtype_nan() -> None:
    check(
        assert_type(pd.array([np.nan], float), NumpyExtensionArray),
        NumpyExtensionArray,
    )
