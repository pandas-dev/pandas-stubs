from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import TimedeltaDtypeArg
from tests.dtypes import ASTYPE_TIMEDELTA_ARGS


@pytest.mark.parametrize(
    "cast_arg, target_type", ASTYPE_TIMEDELTA_ARGS.items(), ids=repr
)
def test_astype_timedelta(cast_arg: TimedeltaDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        assert_type(s.astype("timedelta64[Y]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[M]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[W]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[D]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[h]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[m]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[s]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[ms]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[us]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[μs]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[ns]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[ps]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[fs]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("timedelta64[as]"), "pd.Series[pd.Timedelta]")
        # numpy timedelta64 type codes
        assert_type(s.astype("m8[Y]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[M]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[W]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[D]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[h]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[m]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[s]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[ms]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[us]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[μs]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[ns]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[ps]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[fs]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("m8[as]"), "pd.Series[pd.Timedelta]")
        # numpy timedelta64 type codes
        assert_type(s.astype("<m8[Y]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[M]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[W]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[D]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[h]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[m]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[s]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[ms]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[us]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[μs]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[ns]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[ps]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[fs]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("<m8[as]"), "pd.Series[pd.Timedelta]")
        # pyarrow duration
        assert_type(s.astype("duration[s][pyarrow]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("duration[ms][pyarrow]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("duration[us][pyarrow]"), "pd.Series[pd.Timedelta]")
        assert_type(s.astype("duration[ns][pyarrow]"), "pd.Series[pd.Timedelta]")
