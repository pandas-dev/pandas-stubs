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
from tests._typing import TimestampDtypeArg
from tests.dtypes import (
    ASTYPE_TIMESTAMP_ARGS,
)


def test_series_construction_timestamp_dtype() -> None:
    """Test allowable resolutions for pd.Series() construction with datetime64 dtype.

    Only s, ms, us, ns resolutions are valid for Series construction.
    Resolutions Y, M, W, D, h, m, μs, ps, fs, as are only valid for astype().
    """

    # numpy datetime64: only s, ms, us, ns are valid for construction
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[s]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[ms]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[us]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[ns]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    # numpy datetime64 type codes
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="M8[s]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="M8[ms]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="M8[us]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="M8[ns]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    # little-endian numpy datetime64 type codes
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="<M8[s]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="<M8[ms]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="<M8[us]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Series([datetime.datetime(2020, 1, 1)], dtype="<M8[ns]"),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )

    # PANDAS_UNITS (Y, M, W, D, h, m, μs, ps, fs, as) are not valid for
    # Series construction — only for astype(). Using them gives Series[datetime]
    # instead of Series[Timestamp], indicating the dtype is unsupported.

    if TYPE_CHECKING_INVALID_USAGE:

        def _ts_Y() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[Y]"), Never
            )

        def _ts_M() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[M]"), Never
            )

        def _ts_W() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[W]"), Never
            )

        def _ts_D() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[D]"), Never
            )

        def _ts_h() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[h]"), Never
            )

        def _ts_m() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[m]"), Never
            )

        def _ts_mus() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[μs]"),
                Never,
            )

        def _ts_ps() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[ps]"),
                Never,
            )

        def _ts_fs() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[fs]"),
                Never,
            )

        def _ts_as() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(
                pd.Series([datetime.datetime(2020, 1, 1)], dtype="datetime64[as]"),
                Never,
            )


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
