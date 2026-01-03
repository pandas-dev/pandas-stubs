from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
)
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.core.arrays.datetimes import DatetimeArray
import pytest
from typing_extensions import assert_type

from pandas._libs.tslibs.nattype import NaTType

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    NumpyTimestampDtypeArg,
    PandasTimestampDtypeArg,
)
from tests.dtypes import (
    NUMPY_TIMESTAMP_ARGS,
    PANDAS_TIMESTAMP_ARGS,
)
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize(
    "data", powerset([datetime(2026, 1, 4), pd.Timestamp(2026, 1, 8)], 1)
)
@pytest.mark.parametrize("missing_values", powerset([None, pd.NaT]))
def test_construction_sequence_pandas(
    data: tuple[datetime | pd.Timestamp, ...],
    missing_values: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
) -> None:
    # TODO: pandas-dev/pandas#57064
    # In Pandas 3.0, mixing np.datetime64, datetime and pd.Timestamp also gives DatetimeArray
    check(pd.array(typ([*data, *missing_values])), DatetimeArray)
    check(pd.array(typ([datetime(2077, 1, 1), *data, *missing_values])), DatetimeArray)
    check(
        pd.array(typ([pd.Timestamp(1988, 1, 1), *data, *missing_values])), DatetimeArray
    )

    if TYPE_CHECKING:
        assert_type(pd.array([datetime(1994, 1, 5)]), DatetimeArray)
        assert_type(
            pd.array([pd.Timestamp(2026, 1, 5, tzinfo=ZoneInfo("Africa/Ouagadougou"))]),
            DatetimeArray,
        )

        assert_type(
            pd.array([datetime(2100, 1, 5, 1), datetime(1, 1, 6, 2)]), DatetimeArray
        )
        assert_type(
            pd.array([pd.Timestamp(2002, 1, 5), pd.Timestamp(2, 1, 6)]), DatetimeArray
        )
        assert_type(
            pd.array([datetime(2052, 1, 5), pd.Timestamp(2, 1, 6)]), DatetimeArray
        )

        assert_type(pd.array([datetime(2061, 1, 5, 1), None]), DatetimeArray)
        assert_type(pd.array([pd.Timestamp(1902, 1, 5, 3), None]), DatetimeArray)

        assert_type(pd.array([datetime(1921, 1, 5, 1), pd.NaT]), DatetimeArray)  # type: ignore[assert-type]
        assert_type(pd.array([pd.Timestamp(1872, 1, 5, 3), pd.NaT]), DatetimeArray)  # type: ignore[assert-type]

        assert_type(pd.array([datetime(1751, 1, 5, 1), None, pd.NaT]), DatetimeArray)
        assert_type(
            pd.array([pd.Timestamp(2102, 1, 5, 3), None, pd.NaT]), DatetimeArray
        )

        assert_type(pd.array((datetime(2026, 1, 5),)), DatetimeArray)
        assert_type(pd.array(UserList([pd.Timestamp(2026, 1, 5)])), DatetimeArray)


def test_construction_sequence_numpy() -> None:
    # TODO: pandas-dev/pandas#57064
    # In Pandas 3.0, mixing np.datetime64, datetime and pd.Timestamp also gives DatetimeArray
    check(
        assert_type(pd.array([np.datetime64("2026-01-05 23:27:59")]), DatetimeArray),
        DatetimeArray,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.array([np.datetime64("2131-01-05 01:25"), np.datetime64("1748-01-06")]),
            DatetimeArray,
        ),
        DatetimeArray,
        pd.Timestamp,
    )

    check(
        assert_type(pd.array([np.datetime64("2111-01-05"), None]), DatetimeArray),
        DatetimeArray,
        pd.Timestamp,
    )
    check(
        assert_type(pd.array([np.datetime64("2113-01-05"), pd.NaT]), DatetimeArray),  # type: ignore[assert-type]
        DatetimeArray,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.array([np.datetime64("2114-01-05"), None, pd.NaT]), DatetimeArray
        ),
        DatetimeArray,
        pd.Timestamp,
    )

    check(
        assert_type(pd.array((np.datetime64("1959-01-05"),)), DatetimeArray),
        DatetimeArray,
        pd.Timestamp,
    )
    check(
        assert_type(pd.array(UserList([np.datetime64("1701-01-05")])), DatetimeArray),
        DatetimeArray,
        pd.Timestamp,
    )


@pytest.mark.parametrize("data", powerset([datetime(1710, 10, 10), "2020-11-11 10:00"]))
@pytest.mark.parametrize(
    ("dtype", "target_dtype"),
    (
        PANDAS_TIMESTAMP_ARGS
        | NUMPY_TIMESTAMP_ARGS
        | dict.fromkeys(
            [pd.DatetimeTZDtype(tz=ZoneInfo("America/Caracas")), np.dtype("<M8[ns]")],
            datetime,
        )
    ).items(),
)
def test_construction_dtype(
    data: tuple[datetime | str, ...],
    dtype: (
        PandasTimestampDtypeArg
        | NumpyTimestampDtypeArg
        | pd.DatetimeTZDtype
        | np.dtype[np.datetime64]
    ),
    target_dtype: type,
) -> None:
    dtype_notna = target_dtype if data else None
    check(pd.array([*data], dtype), DatetimeArray, dtype_notna)
    check(
        pd.array([np.datetime64("1748-12-24"), *data], dtype),
        DatetimeArray,
        dtype_notna,
    )

    dtype_na = target_dtype if data else NaTType
    check(pd.array([*data, np.nan], dtype), DatetimeArray, dtype_na)
    check(
        pd.array([np.datetime64("2048-12-24"), *data, np.nan], dtype),
        DatetimeArray,
        target_dtype,
    )

    if TYPE_CHECKING:
        # Pandas datetime64
        assert_type(pd.array(["2001-01-01"], "datetime64[s, UTC]"), DatetimeArray)
        assert_type(pd.array([], "datetime64[us, UTC]"), DatetimeArray)
        assert_type(pd.array([], "datetime64[ns, UTC]"), DatetimeArray)
        assert_type(
            pd.array([], pd.DatetimeTZDtype(tz="Australia/Brisbane")), DatetimeArray
        )

        # Numpy datetime64
        assert_type(pd.array([], "datetime64[s]"), DatetimeArray)
        assert_type(pd.array([], "datetime64[ms]"), DatetimeArray)
        assert_type(pd.array([], "datetime64[us]"), DatetimeArray)
        assert_type(pd.array([], "datetime64[ns]"), DatetimeArray)
        assert_type(pd.array([], "M8[s]"), DatetimeArray)
        assert_type(pd.array([], "M8[ms]"), DatetimeArray)
        assert_type(pd.array([], "M8[us]"), DatetimeArray)
        assert_type(pd.array([], "M8[ns]"), DatetimeArray)
        assert_type(pd.array([], "<M8[s]"), DatetimeArray)
        assert_type(pd.array([], "<M8[ms]"), DatetimeArray)
        assert_type(pd.array([], "<M8[us]"), DatetimeArray)
        assert_type(pd.array([], "<M8[ns]"), DatetimeArray)
        assert_type(pd.array([], np.dtype("<M8[ns]")), DatetimeArray)

    if TYPE_CHECKING_INVALID_USAGE:
        pd.array([], "datetime64[Y]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[M]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[W]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[D]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[h]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[m]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[μs]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[ps]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[fs]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "datetime64[as]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[Y]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[M]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[W]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[D]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[h]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[m]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[μs]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[ps]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[fs]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "M8[as]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[Y]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[M]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[W]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[D]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[h]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[m]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[μs]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[ps]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[fs]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        pd.array([], "<M8[as]")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
