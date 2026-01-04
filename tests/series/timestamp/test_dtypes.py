from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import TimestampDtypeArg
from tests.dtypes import ASTYPE_TIMESTAMP_ARGS


@pytest.mark.parametrize(
    "cast_arg, target_type", ASTYPE_TIMESTAMP_ARGS.items(), ids=repr
)
def test_astype_timestamp(cast_arg: TimestampDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])

    if cast_arg in ("date32[pyarrow]", "date64[pyarrow]"):
        x = pd.Series(pd.date_range("2000-01-01", "2000-02-01"))
        check(x.astype(cast_arg), pd.Series, target_type)
    else:
        check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # numpy datetime64
        assert_type(s.astype("datetime64[Y]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[M]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[W]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[D]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[h]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[m]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[s]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[ms]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[us]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[μs]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[ns]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[ps]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[fs]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("datetime64[as]"), "pd.Series[pd.Timestamp]")
        # numpy datetime64 type codes
        assert_type(s.astype("M8[Y]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[M]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[W]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[D]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[h]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[m]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[s]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[ms]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[us]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[μs]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[ns]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[ps]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[fs]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("M8[as]"), "pd.Series[pd.Timestamp]")
        # numpy datetime64 type codes
        assert_type(s.astype("<M8[Y]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[M]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[W]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[D]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[h]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[m]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[s]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[ms]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[us]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[μs]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[ns]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[ps]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[fs]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("<M8[as]"), "pd.Series[pd.Timestamp]")
        # pyarrow timestamp
        assert_type(s.astype("timestamp[s][pyarrow]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("timestamp[ms][pyarrow]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("timestamp[us][pyarrow]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("timestamp[ns][pyarrow]"), "pd.Series[pd.Timestamp]")
        # pyarrow date
        assert_type(s.astype("date32[pyarrow]"), "pd.Series[pd.Timestamp]")
        assert_type(s.astype("date64[pyarrow]"), "pd.Series[pd.Timestamp]")
