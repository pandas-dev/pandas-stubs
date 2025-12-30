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

        # mypy infers any list with np.nan, which is a float, and types like
        # pd.NA and None to be list[object]
        # It would be quite unusual for user code to be mixing np.nan with the
        # other "NA"-like types.
        assert_type(pd.array([np.nan, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]
        assert_type(pd.array([None, pd.NA]), NumpyExtensionArray)
        assert_type(pd.array([None, pd.NaT]), NumpyExtensionArray)
        assert_type(pd.array([pd.NA, pd.NaT]), NumpyExtensionArray)

        assert_type(pd.array([np.nan, None, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]
        assert_type(pd.array([np.nan, pd.NA, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]
        assert_type(pd.array([None, pd.NA, pd.NaT]), NumpyExtensionArray)

        assert_type(pd.array([np.nan, None, pd.NA, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]

        assert_type(pd.array((np.nan, None, pd.NA, pd.NaT)), NumpyExtensionArray)

        assert_type(pd.array(UserList([np.nan, pd.NA, pd.NaT])), NumpyExtensionArray)  # type: ignore[assert-type]


def test_construction_array_like() -> None:
    data = [1, b"a"]
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
