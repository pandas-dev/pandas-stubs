import numpy as np
import pandas as pd
from pandas.arrays import (
    BooleanArray,
    DatetimeArray,
    IntegerArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    StringArray,
    TimedeltaArray,
)

from pandas.tseries.offsets import Day

LIST_MASK = [False, True, False, False, False, False, False, False, True, False]
ARRAY_MASK = np.array(LIST_MASK)


def test_integer_array() -> None:
    ints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    IntegerArray(ints, mask=ARRAY_MASK)
    IntegerArray(ints, mask=ARRAY_MASK, copy=True)

    nulled_ints = [1, 2, 3, 4, 5, 6, 7, 8, None, 10]
    pd.array(nulled_ints, dtype="UInt8")
    pd.array(nulled_ints, dtype=pd.UInt8Dtype())
    pd.array(nulled_ints, dtype=float)
    pd.array(ints, dtype=int)


def test_string_array() -> None:
    strings = np.array(["a", "b", "c", "d", "e", "f", "g", "h", None, "j"])
    StringArray(strings, copy=False)
    StringArray(strings, copy=True)

    strings_list = strings.tolist()
    pd.array(strings_list, dtype="string")
    pd.array(strings, dtype="string")
    pd.array(strings, dtype=str)
    pd.array(strings)


def test_boolean_array() -> None:
    bools = np.array([True, False, True, False, True, False, True, False, True, False])
    BooleanArray(bools, mask=ARRAY_MASK)
    BooleanArray(bools, mask=ARRAY_MASK, copy=True)

    nulled_bools = [True, False, True, False, True, False, True, False, None, False]
    pd.array(nulled_bools)
    pd.array(nulled_bools, dtype="bool")
    pd.array(nulled_bools, dtype=bool)
    pd.array(nulled_bools, dtype=pd.BooleanDtype())


def test_period_array() -> None:
    pa = PeriodArray(
        pd.Series(
            [pd.Period("2000-01-01", freq="D"), pd.Period("2000-01-02", freq="D")]
        )
    )
    PeriodArray(
        pd.Index([pd.Period("2000-01-01", freq="D"), pd.Period("2000-01-02", freq="D")])
    )
    PeriodArray(
        np.ndarray(
            [
                0,
                1,
                2,
            ]
        ),
        freq="D",
    )
    PeriodArray(np.ndarray([0, 1, 2]), freq=Day())
    PeriodArray(pa)
    dt = pd.PeriodDtype(freq="D")
    PeriodArray(
        pd.Index([pd.Period("2000-01-01"), pd.Period("2000-01-02")]),
        dtype=dt,
        copy=False,
    )
    PeriodArray(
        pd.Index([pd.Period("2000-01-01"), pd.Period("2000-01-02")]),
        dtype=dt,
        freq="D",
        copy=False,
    )


def test_datetime_array() -> None:
    values = [pd.Timestamp("2000-1-1"), pd.Timestamp("2000-1-2")]
    DatetimeArray(pd.Index(values), dtype=np.dtype("M8[ns]"), freq="D", copy=False)
    DatetimeArray(pd.Series(values), dtype=np.dtype("M8[ns]"), freq="D", copy=False)
    np_values = np.array([np.datetime64(1, "ns"), np.datetime64(2, "ns")])
    dta = DatetimeArray(np_values)
    # TODO: How to verify DatetimeTZDtype
    # tz =pd.DatetimeTZDtype(tz=None)
    # DatetimeArray(pd.Index(values),dtype=tz, freq=Day(), copy=False)


def test_interval_array() -> None:
    ia = IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)])
    IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="left")
    IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="right")
    IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="both")
    IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="neither")
    IntervalArray(
        [pd.Interval(0, 1), pd.Interval(1, 2)], closed="neither", verify_integrity=True
    )

    ia.left
    ia.right
    ia.closed
    ia.mid
    ia.length
    ia.is_empty
    ia.is_non_overlapping_monotonic

    IntervalArray.from_arrays([0, 1], [1, 2])
    IntervalArray.from_arrays(np.array([0, 1]), np.array([1, 2]))
    IntervalArray.from_arrays(pd.Series([0, 1]), pd.Series([1, 2]))
    IntervalArray.from_arrays(pd.Index([0, 1]), pd.Index([1, 2]))
    IntervalArray.from_arrays([0, 1], [1, 2], closed="left", copy=False)
    IntervalArray.from_arrays(
        [0, 1], [1, 2], closed="right", dtype=pd.IntervalDtype("int64")
    )
    IntervalArray.from_arrays(
        [0, 1], [1, 2], closed="right", dtype=pd.IntervalDtype(float)
    )
    IntervalArray.from_arrays([0, 1], [1, 2], closed="both")
    IntervalArray.from_arrays([0, 1], [1, 2], closed="neither")

    breaks = [0, 1, 2, 3, 4.5]
    IntervalArray.from_breaks(breaks)
    IntervalArray.from_breaks(np.array(breaks), copy=False)
    IntervalArray.from_breaks(pd.Series(breaks), closed="left")
    IntervalArray.from_breaks(pd.Index(breaks), closed="right")
    IntervalArray.from_breaks(pd.Index(breaks), closed="both")
    IntervalArray.from_breaks(pd.Index(breaks), closed="neither")
    IntervalArray.from_breaks(pd.Index(breaks), dtype=pd.IntervalDtype(float))

    ia.contains(0.5)
    ia.overlaps(pd.Interval(0.5, 1.0))
    ia.set_closed("right")
    ia.set_closed("left")
    ia.set_closed("both")
    ia.set_closed("neither")
    ia.to_tuples(True)
    ia.to_tuples(False)


def test_timedelta_array() -> None:
    tda = TimedeltaArray(np.array([1, 2], dtype="timedelta64[ns]"))
    tda = TimedeltaArray(np.array([1, 2], dtype="timedelta64[ns]"), copy=False)
    TimedeltaArray(
        pd.Series([pd.Timedelta("1 days"), pd.Timedelta("2 days")]), freq="D"
    )
    TimedeltaArray(
        pd.Series([pd.Timedelta("1 days"), pd.Timedelta("2 days")]), freq=Day()
    )
    TimedeltaArray(pd.Index([pd.Timedelta("1 days"), pd.Timedelta("2 days")]))
    TimedeltaArray(tda)

    TimedeltaArray(
        pd.Series([pd.Timedelta("1 days"), pd.Timedelta("2 days")]),
        dtype=np.dtype("timedelta64[ns]"),
    )
    TimedeltaArray(
        pd.Series([pd.Timedelta("1 days"), pd.Timedelta("2 days")]),
        dtype=np.dtype("timedelta64[ns]"),
    )


def test_sparse_array() -> None:

    SparseArray(3.0, fill_value=np.nan)
    SparseArray([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], fill_value=np.nan)
    sa = SparseArray([0, 2, 3, 4, 5, 6, 0, 8, 9, 0], fill_value=0)
    SparseArray(sa.sp_values, sparse_index=sa.sp_index)
    sa_block = SparseArray([0, 2, 3, 4, 5, 6, 0, 8, 9, 0], fill_value=0, kind="block")
    SparseArray(sa_block.sp_values, sparse_index=sa_block.sp_index)

    SparseArray(
        [True, False, False, False, False, False, False, True, False, False],
        fill_value=False,
    )
    SparseArray(
        [
            pd.Timestamp("2011-01-01"),
            pd.Timestamp("2011-01-02"),
            pd.Timestamp("2011-01-03"),
            pd.NaT,
            pd.NaT,
        ],
        fill_value=pd.NaT,
    )
    SparseArray([pd.Timedelta(days=1), pd.NaT, pd.NaT], fill_value=pd.NaT)

    SparseArray([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], kind="integer", copy=False)
    SparseArray([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], kind="block", copy=True)
    SparseArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype="i4")
    SparseArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype="int32")
    SparseArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int16)
