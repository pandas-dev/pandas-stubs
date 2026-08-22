from __future__ import annotations

import datetime
from typing import (
    TYPE_CHECKING,
    Never,
    assert_type,
)

import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    TimedeltaDtypeArg,
)
from tests.dtypes import ASTYPE_TIMEDELTA_ARGS


def test_series_construction_timedelta_dtype() -> None:
    """
    Test allowable resolutions for pd.Series() construction with timedelta64 dtype.

    Only s, ms, us, ns resolutions are valid for Series construction.
    Resolutions Y, M, W, D, h, m, μs, ps, fs, as are only valid for astype().
    """

    # numpy timedelta64: only s, ms, us, ns are valid for construction
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[s]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[ms]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[us]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[ns]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    # numpy timedelta64 type codes
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="m8[s]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="m8[ms]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="m8[us]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="m8[ns]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    # little-endian numpy timedelta64 type codes
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="<m8[s]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="<m8[ms]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="<m8[us]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.Series([datetime.timedelta(seconds=1)], dtype="<m8[ns]"),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )

    # PANDAS_UNITS (Y, M, W, D, h, m, μs, ps, fs, as) are not valid for
    # Series construction — only for astype(). Using them gives Series[timedelta]
    # instead of Series[Timedelta], indicating the dtype is unsupported.

    if TYPE_CHECKING_INVALID_USAGE:

        def _td_Y() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[Y]"),
                Never,
            )

        def _td_M() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[M]"),
                Never,
            )

        def _td_W() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[W]"),
                Never,
            )

        def _td_D() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[D]"),
                Never,
            )

        def _td_h() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[h]"),
                Never,
            )

        def _td_m() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[m]"),
                Never,
            )

        def _td_mus() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[μs]"),
                Never,
            )

        def _td_ps() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[ps]"),
                Never,
            )

        def _td_fs() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[fs]"),
                Never,
            )

        def _td_as() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.timedelta(seconds=1)], dtype="timedelta64[as]"),
                Never,
            )


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
