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
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    PD_LTE_23,
    check,
)
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("missing_values", powerset([None, pd.NA, pd.NaT], 1))
def test_construction_sequence(
    missing_values: tuple[Any, ...], typ: Callable[[Sequence[Any]], Sequence[Any]]
) -> None:
    # `pd.NaT in [pd.NA, pd.NaT]` leads to an exception
    if missing_values[-1] is pd.NaT:
        expected_type = DatetimeArray if PD_LTE_23 else NumpyExtensionArray
        check(pd.array(typ(missing_values)), expected_type)
        check(pd.array(typ((np.nan, *missing_values))), expected_type)
    else:
        check(pd.array(typ(missing_values)), NumpyExtensionArray)

    if TYPE_CHECKING:
        assert_type(pd.array([None]), NumpyExtensionArray)
        assert_type(pd.array([pd.NA]), NumpyExtensionArray)
        assert_type(pd.array([pd.NaT]), NumpyExtensionArray)

        # mypy does not like the literal version pd.array([np.nan, pd.NaT])
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

        assert_type(pd.array((np.nan, None, pd.NA, pd.NaT)), NumpyExtensionArray)

        _50 = UserList([np.nan, pd.NA, pd.NaT])
        assert_type(pd.array(_50), NumpyExtensionArray)


def test_construction_array_like() -> None:
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


def test_construction_dtype_nan() -> None:
    check(
        assert_type(pd.array([np.nan], float), NumpyExtensionArray),
        NumpyExtensionArray,
    )
