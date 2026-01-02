from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import (
    LINUX,
    MAC,
    check,
)
from tests._typing import IntDtypeArg
from tests.dtypes import (
    ASTYPE_INT_ARGS,
    ASTYPE_UINT_ARGS,
)


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_INT_ARGS.items(), ids=repr)
def test_astype_int(cast_arg: IntDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])

    s_astype = s.astype(cast_arg)
    if (LINUX or MAC) and cast_arg in {np.longlong, "longlong", "q"}:
        # TODO: pandas-dev/pandas#54252 longlong is bugged on Linux and Mac
        msg = rf"Expected type '{target_type}' but got '{type(s_astype.iloc[0])}'"
        with pytest.raises(RuntimeError, match=msg):
            check(s_astype, pd.Series, target_type)
    else:
        check(s_astype, pd.Series, target_type)

    if TYPE_CHECKING:
        # python int
        assert_type(s.astype(int), "pd.Series[int]")
        assert_type(s.astype("int"), "pd.Series[int]")
        # pandas Int8
        assert_type(s.astype(pd.Int8Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int8"), "pd.Series[int]")
        # pandas Int16
        assert_type(s.astype(pd.Int16Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int16"), "pd.Series[int]")
        # pandas Int32
        assert_type(s.astype(pd.Int32Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int32"), "pd.Series[int]")
        # pandas Int64
        assert_type(s.astype(pd.Int64Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int64"), "pd.Series[int]")
        # numpy int8
        assert_type(s.astype(np.byte), "pd.Series[int]")
        assert_type(s.astype("byte"), "pd.Series[int]")
        assert_type(s.astype("int8"), "pd.Series[int]")
        assert_type(s.astype("b"), "pd.Series[int]")
        assert_type(s.astype("i1"), "pd.Series[int]")
        # numpy int16
        assert_type(s.astype(np.short), "pd.Series[int]")
        assert_type(s.astype("short"), "pd.Series[int]")
        assert_type(s.astype("int16"), "pd.Series[int]")
        assert_type(s.astype("h"), "pd.Series[int]")
        assert_type(s.astype("i2"), "pd.Series[int]")
        # numpy int32
        assert_type(s.astype(np.intc), "pd.Series[int]")
        assert_type(s.astype("intc"), "pd.Series[int]")
        assert_type(s.astype("int32"), "pd.Series[int]")
        assert_type(s.astype("i"), "pd.Series[int]")
        assert_type(s.astype("i4"), "pd.Series[int]")
        # numpy int64
        assert_type(s.astype(np.int_), "pd.Series[int]")
        assert_type(s.astype("int_"), "pd.Series[int]")
        assert_type(s.astype("int64"), "pd.Series[int]")
        assert_type(s.astype("long"), "pd.Series[int]")
        assert_type(s.astype("l"), "pd.Series[int]")
        assert_type(s.astype("i8"), "pd.Series[int]")
        # numpy signed pointer
        assert_type(s.astype(np.intp), "pd.Series[int]")
        assert_type(s.astype("intp"), "pd.Series[int]")
        assert_type(s.astype("p"), "pd.Series[int]")
        # pyarrow integer types
        assert_type(s.astype("int8[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("int16[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("int32[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("int64[pyarrow]"), "pd.Series[int]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_UINT_ARGS.items(), ids=repr)
def test_astype_uint(cast_arg: IntDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # pandas UInt8
        assert_type(s.astype(pd.UInt8Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt8"), "pd.Series[int]")
        # pandas UInt16
        assert_type(s.astype(pd.UInt16Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt16"), "pd.Series[int]")
        # pandas UInt32
        assert_type(s.astype(pd.UInt32Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt32"), "pd.Series[int]")
        # pandas UInt64
        assert_type(s.astype(pd.UInt64Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt64"), "pd.Series[int]")
        # numpy uint8
        assert_type(s.astype(np.ubyte), "pd.Series[int]")
        assert_type(s.astype("ubyte"), "pd.Series[int]")
        assert_type(s.astype("uint8"), "pd.Series[int]")
        assert_type(s.astype("B"), "pd.Series[int]")
        assert_type(s.astype("u1"), "pd.Series[int]")
        # numpy uint16
        assert_type(s.astype(np.ushort), "pd.Series[int]")
        assert_type(s.astype("ushort"), "pd.Series[int]")
        assert_type(s.astype("uint16"), "pd.Series[int]")
        assert_type(s.astype("H"), "pd.Series[int]")
        assert_type(s.astype("u2"), "pd.Series[int]")
        # numpy uint32
        assert_type(s.astype(np.uintc), "pd.Series[int]")
        assert_type(s.astype("uintc"), "pd.Series[int]")
        assert_type(s.astype("uint32"), "pd.Series[int]")
        assert_type(s.astype("I"), "pd.Series[int]")
        assert_type(s.astype("u4"), "pd.Series[int]")
        # numpy uint64
        assert_type(s.astype(np.uint), "pd.Series[int]")
        assert_type(s.astype("uint"), "pd.Series[int]")
        assert_type(s.astype("uint64"), "pd.Series[int]")
        assert_type(s.astype("ulong"), "pd.Series[int]")
        assert_type(s.astype("L"), "pd.Series[int]")
        assert_type(s.astype("u8"), "pd.Series[int]")
        # numpy unsigned pointer
        assert_type(s.astype(np.uintp), "pd.Series[int]")
        assert_type(s.astype("uintp"), "pd.Series[int]")
        assert_type(s.astype("P"), "pd.Series[int]")
        # pyarrow unsigned integer types
        assert_type(s.astype("uint8[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("uint16[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("uint32[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("uint64[pyarrow]"), "pd.Series[int]")
