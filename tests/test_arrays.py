from typing import Type

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.arrays import (
    BooleanArray,
    DatetimeArray,
    IntegerArray,
    IntervalArray,
    PandasArray,
    PeriodArray,
    SparseArray,
    StringArray,
    TimedeltaArray,
)
from pandas.core.arrays.base import ExtensionArray
from typing_extensions import assert_type

from pandas._libs.sparse import (
    BlockIndex,
    IntIndex,
    SparseIndex,
)

from tests import check

from pandas.tseries.offsets import Day

LIST_MASK = [False, True, False, False, False, False, False, False, True, False]
ARRAY_MASK = np.array(LIST_MASK)


def test_integer_array() -> None:
    ints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    non_null_int_arr = IntegerArray(ints, mask=np.array([False] * 10))
    int_arr = IntegerArray(ints, mask=ARRAY_MASK)
    check(assert_type(int_arr, IntegerArray), IntegerArray)
    check(
        assert_type(IntegerArray(ints, mask=ARRAY_MASK, copy=True), IntegerArray),
        IntegerArray,
    )

    nulled_ints = [1, 2, 3, 4, 5, 6, 7, 8, None, 10]
    check(
        assert_type(pd.array(nulled_ints, dtype="UInt8"), Type[ExtensionArray]),
        IntegerArray,
    )
    check(
        assert_type(pd.array(nulled_ints, dtype=pd.UInt8Dtype()), Type[ExtensionArray]),
        IntegerArray,
    )
    check(
        assert_type(pd.array(nulled_ints, dtype=float), Type[ExtensionArray]),
        PandasArray,
    )
    check(assert_type(pd.array(ints, dtype=int), Type[ExtensionArray]), PandasArray)

    check(assert_type(int_arr.astype("Int64"), IntegerArray), IntegerArray)
    check(assert_type(int_arr.astype("UInt8"), IntegerArray), IntegerArray)
    check(
        assert_type(
            int_arr.astype(pd.BooleanDtype()),
            BooleanArray,
        ),
        BooleanArray,
    )
    check(assert_type(int_arr.astype(float), npt.NDArray[np.float64]), np.ndarray)
    check(assert_type(int_arr.astype(np.float64), npt.NDArray[np.float64]), np.ndarray)
    non_null_int_arr.astype(bool)
    non_null_int_arr.astype(np.bool_)
    non_null_int_arr.astype("bool")

    non_null_int_arr.astype(int)
    non_null_int_arr.astype("i1")
    non_null_int_arr.astype("i2")
    non_null_int_arr.astype("i4")
    non_null_int_arr.astype("i8")
    non_null_int_arr.astype("int8")
    non_null_int_arr.astype("int16")
    non_null_int_arr.astype("int32")
    non_null_int_arr.astype("int64")
    non_null_int_arr.astype(np.int8)
    non_null_int_arr.astype(np.int16)
    non_null_int_arr.astype(np.int32)
    non_null_int_arr.astype(np.int64)

    non_null_int_arr.astype("u1")
    non_null_int_arr.astype("u2")
    non_null_int_arr.astype("u4")
    non_null_int_arr.astype("u8")
    non_null_int_arr.astype("uint8")
    non_null_int_arr.astype("uint16")
    non_null_int_arr.astype("uint32")
    non_null_int_arr.astype("uint64")
    non_null_int_arr.astype(np.uint8)
    non_null_int_arr.astype(np.uint16)
    non_null_int_arr.astype(np.uint32)
    non_null_int_arr.astype(np.uint64)

    non_null_int_arr.astype(np.float32)
    non_null_int_arr.astype("float32")

    non_null_int_arr.astype(float)
    non_null_int_arr.astype("float")
    non_null_int_arr.astype("float64")
    non_null_int_arr.astype(np.float64)

    non_null_int_arr.astype(np.complex64)
    non_null_int_arr.astype("float64")

    non_null_int_arr.astype("c8")
    non_null_int_arr.astype("complex64")
    non_null_int_arr.astype(np.complex128)

    non_null_int_arr.astype(complex)
    non_null_int_arr.astype("complex")
    non_null_int_arr.astype("c16")
    non_null_int_arr.astype("complex128")
    non_null_int_arr.astype(np.complex128)

    non_null_int_arr.astype("M8[ns]")
    non_null_int_arr.astype(np.datetime64)

    non_null_int_arr.astype(str)

    int_arr.astype("boolean")
    int_arr.astype(pd.BooleanDtype())

    int_arr.astype("Int8")
    int_arr.astype("Int16")
    int_arr.astype("Int32")
    int_arr.astype("Int64")
    int_arr.astype("UInt8")
    int_arr.astype("UInt16")
    int_arr.astype("UInt32")
    int_arr.astype("UInt64")
    int_arr.astype(pd.Int8Dtype())
    int_arr.astype(pd.Int16Dtype())
    int_arr.astype(pd.Int32Dtype())
    int_arr.astype(pd.Int64Dtype())
    int_arr.astype(pd.UInt8Dtype())
    int_arr.astype(pd.UInt16Dtype())
    int_arr.astype(pd.UInt32Dtype())
    int_arr.astype(pd.UInt64Dtype())

    int_arr.astype("string")
    int_arr.astype(pd.StringDtype())

    int_arr.astype(pd.DatetimeTZDtype(tz="UTC"))

    # TODO: Test get/set item


def test_string_array() -> None:
    strings = np.array(["a", "b", "c", "d", "e", "f", "g", "h", None, "j"])
    check(assert_type(StringArray(strings, copy=False), StringArray), StringArray)
    check(assert_type(StringArray(strings, copy=True), StringArray), StringArray)

    strings_list = strings.tolist()
    check(
        assert_type(pd.array(strings_list, dtype="string"), Type[ExtensionArray]),
        StringArray,
    )
    check(
        assert_type(pd.array(strings, dtype="string"), Type[ExtensionArray]),
        StringArray,
    )
    check(assert_type(pd.array(strings, dtype=str), Type[ExtensionArray]), PandasArray)
    check(assert_type(pd.array(strings), Type[ExtensionArray]), StringArray)


def test_boolean_array() -> None:
    bools = np.array([True, False, True, False, True, False, True, False, True, False])
    check(assert_type(BooleanArray(bools, mask=ARRAY_MASK), BooleanArray), BooleanArray)
    check(
        assert_type(BooleanArray(bools, mask=ARRAY_MASK, copy=True), BooleanArray),
        BooleanArray,
    )

    nulled_bools = [True, False, True, False, True, False, True, False, None, False]
    check(assert_type(pd.array(nulled_bools), Type[ExtensionArray]), BooleanArray)
    check(
        assert_type(pd.array(nulled_bools, dtype="bool"), Type[ExtensionArray]),
        PandasArray,
    )
    check(
        assert_type(pd.array(nulled_bools, dtype=bool), Type[ExtensionArray]),
        PandasArray,
    )
    check(
        assert_type(
            pd.array(nulled_bools, dtype=pd.BooleanDtype()), Type[ExtensionArray]
        ),
        BooleanArray,
    )


def test_period_array() -> None:
    p1 = pd.Period("2000-01-01", freq="D")
    p2 = pd.Period("2000-01-02", freq="D")
    pa = PeriodArray(pd.Series([p1, p2]))
    check(assert_type(pa, PeriodArray), PeriodArray)
    check(assert_type(PeriodArray(pd.Index([p1, p2])), PeriodArray), PeriodArray)
    int_arr: npt.NDArray[np.int_] = np.ndarray([0, 1, 2])
    check(assert_type(PeriodArray(int_arr, freq="D"), PeriodArray), PeriodArray)
    check(
        assert_type(PeriodArray(np.ndarray([0, 1, 2]), freq=Day()), PeriodArray),
        PeriodArray,
    )
    check(assert_type(PeriodArray(pa), PeriodArray), PeriodArray)
    dt = pd.PeriodDtype(freq="D")
    period_idx = pd.Index([p1, p2])
    check(
        assert_type(PeriodArray(period_idx, dtype=dt, copy=False), PeriodArray),
        PeriodArray,
    )

    check(
        assert_type(
            PeriodArray(period_idx, dtype=dt, freq="D", copy=False), PeriodArray
        ),
        PeriodArray,
    )

    check(assert_type(pd.array([p1, p2]), Type[ExtensionArray]), PeriodArray)
    check(
        assert_type(pd.array([p1, p2], dtype="period[D]"), Type[ExtensionArray]),
        PeriodArray,
    )


def test_datetime_array() -> None:
    values = [pd.Timestamp("2000-1-1"), pd.Timestamp("2000-1-2")]
    check(
        assert_type(
            DatetimeArray(
                pd.Index(values), dtype=np.dtype("M8[ns]"), freq="D", copy=False
            ),
            DatetimeArray,
        ),
        DatetimeArray,
    )
    check(
        assert_type(
            DatetimeArray(
                pd.Series(values), dtype=np.dtype("M8[ns]"), freq="D", copy=False
            ),
            DatetimeArray,
        ),
        DatetimeArray,
    )
    np_values = np.array([np.datetime64(1, "ns"), np.datetime64(2, "ns")])
    dta = DatetimeArray(np_values)
    check(assert_type(DatetimeArray(dta), DatetimeArray), DatetimeArray)
    data = np.array([1, 2, 3], dtype="M8[ns]")
    check(
        assert_type(
            DatetimeArray(data, copy=False, dtype=pd.DatetimeTZDtype(tz="US/Central")),
            DatetimeArray,
        ),
        DatetimeArray,
    )

    check(assert_type(pd.array(data), Type[ExtensionArray]), DatetimeArray)
    check(assert_type(pd.array(np_values), Type[ExtensionArray]), DatetimeArray)


def test_interval_array_construction() -> None:
    ia = IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)])
    check(assert_type(IntervalArray(ia), IntervalArray), IntervalArray)
    check(
        assert_type(
            IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="left"),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="right"),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="both"),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)], closed="neither"),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray(
                [pd.Interval(0, 1), pd.Interval(1, 2)],
                closed="neither",
                verify_integrity=True,
            ),
            IntervalArray,
        ),
        IntervalArray,
    )

    check(
        assert_type(IntervalArray.from_arrays([0, 1], [1, 2]), IntervalArray),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays(np.array([0, 1]), np.array([1, 2])), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays(pd.Series([0, 1]), pd.Series([1, 2])),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays(pd.Index([0, 1]), pd.Index([1, 2])), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays([0, 1], [1, 2], closed="left", copy=False),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays(
                [0, 1], [1, 2], closed="right", dtype=pd.IntervalDtype("int64")
            ),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays(
                [0, 1], [1, 2], closed="right", dtype=pd.IntervalDtype(float)
            ),
            IntervalArray,
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays([0, 1], [1, 2], closed="both"), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_arrays([0, 1], [1, 2], closed="neither"), IntervalArray
        ),
        IntervalArray,
    )

    breaks = [0, 1, 2, 3, 4.5]
    check(assert_type(IntervalArray.from_breaks(breaks), IntervalArray), IntervalArray)
    check(
        assert_type(
            IntervalArray.from_breaks(np.array(breaks), copy=False), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_breaks(pd.Series(breaks), closed="left"), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_breaks(pd.Index(breaks), closed="right"), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_breaks(pd.Index(breaks), closed="both"), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_breaks(pd.Index(breaks), closed="neither"), IntervalArray
        ),
        IntervalArray,
    )
    check(
        assert_type(
            IntervalArray.from_breaks(pd.Index(breaks), dtype=pd.IntervalDtype(float)),
            IntervalArray,
        ),
        IntervalArray,
    )


def test_integer_array_attrib_props() -> None:
    ia = IntervalArray([pd.Interval(0, 1), pd.Interval(1, 2)])

    check(assert_type(ia.left, pd.Int64Index), pd.Int64Index)
    check(assert_type(ia.right, pd.Int64Index), pd.Int64Index)
    check(assert_type(ia.closed, str), str)
    check(assert_type(ia.mid, pd.Float64Index), pd.Float64Index)
    check(assert_type(ia.length, pd.Int64Index), pd.Int64Index)
    check(assert_type(ia.is_empty, npt.NDArray[np.bool_]), np.ndarray)
    check(assert_type(ia.is_non_overlapping_monotonic, bool), bool)

    check(assert_type(ia.contains(0.5), npt.NDArray[np.bool_]), np.ndarray)
    check(
        assert_type(ia.overlaps(pd.Interval(0.5, 1.0)), npt.NDArray[np.bool_]),
        np.ndarray,
    )
    check(assert_type(ia.set_closed("right"), IntervalArray), IntervalArray)
    check(assert_type(ia.set_closed("left"), IntervalArray), IntervalArray)
    check(assert_type(ia.set_closed("both"), IntervalArray), IntervalArray)
    check(assert_type(ia.set_closed("neither"), IntervalArray), IntervalArray)
    check(assert_type(ia.to_tuples(True), npt.NDArray[np.object_]), np.ndarray)
    check(assert_type(ia.to_tuples(False), npt.NDArray[np.object_]), np.ndarray)

    ia_float = IntervalArray([pd.Interval(0, 1.5), pd.Interval(1, 2)])
    check(assert_type(ia_float.left, pd.Float64Index), pd.Float64Index)
    check(assert_type(ia_float.right, pd.Float64Index), pd.Float64Index)
    check(assert_type(ia_float.length, pd.Float64Index), pd.Float64Index)

    ia_ts = IntervalArray(
        [
            pd.Interval(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02")),
            pd.Interval(pd.Timestamp("2018-01-02"), pd.Timestamp("2018-01-03")),
        ]
    )
    check(assert_type(ia_ts.left, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(ia_ts.right, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(ia_ts.mid, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(ia_ts.length, pd.TimedeltaIndex), pd.TimedeltaIndex)

    ia_td = IntervalArray(
        [
            pd.Interval(pd.Timedelta("1 days"), pd.Timedelta("2 days")),
            pd.Interval(pd.Timedelta("2 days"), pd.Timedelta("3 days")),
        ]
    )
    check(assert_type(ia_td.left, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(ia_td.right, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(ia_td.mid, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(ia_td.length, pd.TimedeltaIndex), pd.TimedeltaIndex)


def test_timedelta_array() -> None:
    td1, td2 = pd.Timedelta("1 days"), pd.Timedelta("2 days")
    tda = TimedeltaArray(np.array([1, 2], dtype="timedelta64[ns]"))
    check(assert_type(tda, TimedeltaArray), TimedeltaArray)

    tda = TimedeltaArray(np.array([1, 2], dtype="timedelta64[ns]"), copy=False)
    tds = pd.Series([td1, td2])
    tdi = pd.Index([td1, td2])

    check(assert_type(tda, TimedeltaArray), TimedeltaArray)
    check(assert_type(TimedeltaArray(tds, freq="D"), TimedeltaArray), TimedeltaArray)
    check(assert_type(TimedeltaArray(tds, freq=Day()), TimedeltaArray), TimedeltaArray)
    check(assert_type(TimedeltaArray(tdi), TimedeltaArray), TimedeltaArray)
    check(assert_type(TimedeltaArray(tda), TimedeltaArray), TimedeltaArray)

    check(
        assert_type(
            TimedeltaArray(tds, dtype=np.dtype("timedelta64[ns]")), TimedeltaArray
        ),
        TimedeltaArray,
    )
    check(
        assert_type(
            TimedeltaArray(tds, dtype=np.dtype("timedelta64[ns]")), TimedeltaArray
        ),
        TimedeltaArray,
    )

    check(
        assert_type(
            pd.array(np.array([1, 2], dtype="timedelta64[ns]")), Type[ExtensionArray]
        ),
        TimedeltaArray,
    )
    check(assert_type(pd.array(tdi), Type[ExtensionArray]), TimedeltaArray)
    check(assert_type(pd.array(tds, copy=False), Type[ExtensionArray]), TimedeltaArray)


def test_sparse_array() -> None:
    ints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    nulled_ints = [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]
    zero_ints = [0, 2, 3, 4, 5, 6, 0, 8, 9, 0]

    check(assert_type(SparseArray(3.0, fill_value=np.nan), SparseArray), SparseArray)
    check(
        assert_type(
            SparseArray(nulled_ints, fill_value=np.nan),
            SparseArray,
        ),
        SparseArray,
    )
    sa = SparseArray(zero_ints, fill_value=0)
    check(assert_type(sa.sp_index, SparseIndex), IntIndex)
    check(
        assert_type(SparseArray(sa.sp_values, sparse_index=sa.sp_index), SparseArray),
        SparseArray,
    )
    sa_block = SparseArray(zero_ints, fill_value=0, kind="block")
    check(assert_type(sa_block.sp_index, SparseIndex), BlockIndex)
    check(
        assert_type(
            SparseArray(sa_block.sp_values, sparse_index=sa_block.sp_index), SparseArray
        ),
        SparseArray,
    )

    check(
        assert_type(
            SparseArray(
                [True, False, False, False, False, False, False, True, False, False],
                fill_value=False,
            ),
            SparseArray,
        ),
        SparseArray,
    )
    check(
        assert_type(
            SparseArray(
                [
                    pd.Timestamp("2011-01-01"),
                    pd.Timestamp("2011-01-02"),
                    pd.Timestamp("2011-01-03"),
                    pd.NaT,
                    pd.NaT,
                ],
                fill_value=pd.NaT,
            ),
            SparseArray,
        ),
        SparseArray,
    )
    check(
        assert_type(
            SparseArray([pd.Timedelta(days=1), pd.NaT, pd.NaT], fill_value=pd.NaT),
            SparseArray,
        ),
        SparseArray,
    )

    check(
        assert_type(
            SparseArray(nulled_ints, kind="integer", copy=False),
            SparseArray,
        ),
        SparseArray,
    )
    check(
        assert_type(
            SparseArray(nulled_ints, kind="block", copy=True),
            SparseArray,
        ),
        SparseArray,
    )
    check(assert_type(SparseArray(ints, dtype="i4"), SparseArray), SparseArray)
    check(assert_type(SparseArray(ints, dtype="int32"), SparseArray), SparseArray)
    check(assert_type(SparseArray(ints, dtype=np.int16), SparseArray), SparseArray)
