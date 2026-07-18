from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    assert_type,
)

import numpy as np
import pandas as pd
import pytest

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
    i = pd.Index([1, 2, 3])

    s_astype = i.astype(cast_arg)
    if (LINUX or MAC) and cast_arg in {np.longlong, "longlong", "q"}:
        # TODO: pandas-dev/pandas#54252 longlong is bugged on Linux and Mac
        msg = rf"Expected type '{target_type}' but got '{type(s_astype[0])}'"
        with pytest.raises(RuntimeError, match=msg):
            check(s_astype, pd.Index, target_type)
    else:
        check(s_astype, pd.Index, target_type)

    if TYPE_CHECKING:
        # python int
        assert_type(i.astype(int), "pd.Index[int]")
        assert_type(i.astype("int"), "pd.Index[int]")
        # pandas Int8
        assert_type(i.astype(pd.Int8Dtype()), "pd.Index[int]")
        assert_type(i.astype("Int8"), "pd.Index[int]")
        # pandas Int16
        assert_type(i.astype(pd.Int16Dtype()), "pd.Index[int]")
        assert_type(i.astype("Int16"), "pd.Index[int]")
        # pandas Int32
        assert_type(i.astype(pd.Int32Dtype()), "pd.Index[int]")
        assert_type(i.astype("Int32"), "pd.Index[int]")
        # pandas Int64
        assert_type(i.astype(pd.Int64Dtype()), "pd.Index[int]")
        assert_type(i.astype("Int64"), "pd.Index[int]")
        # numpy int8
        assert_type(i.astype(np.byte), "pd.Index[int]")
        assert_type(i.astype("byte"), "pd.Index[int]")
        assert_type(i.astype("int8"), "pd.Index[int]")
        assert_type(i.astype("b"), "pd.Index[int]")
        assert_type(i.astype("i1"), "pd.Index[int]")
        # numpy int16
        assert_type(i.astype(np.short), "pd.Index[int]")
        assert_type(i.astype("short"), "pd.Index[int]")
        assert_type(i.astype("int16"), "pd.Index[int]")
        assert_type(i.astype("h"), "pd.Index[int]")
        assert_type(i.astype("i2"), "pd.Index[int]")
        # numpy int32
        assert_type(i.astype(np.intc), "pd.Index[int]")
        assert_type(i.astype("intc"), "pd.Index[int]")
        assert_type(i.astype("int32"), "pd.Index[int]")
        assert_type(i.astype("i"), "pd.Index[int]")
        assert_type(i.astype("i4"), "pd.Index[int]")
        # numpy int64
        assert_type(i.astype(np.int_), "pd.Index[int]")
        assert_type(i.astype("int_"), "pd.Index[int]")
        assert_type(i.astype("int64"), "pd.Index[int]")
        assert_type(i.astype("long"), "pd.Index[int]")
        assert_type(i.astype("l"), "pd.Index[int]")
        assert_type(i.astype("i8"), "pd.Index[int]")
        # numpy signed pointer
        assert_type(i.astype(np.intp), "pd.Index[int]")
        assert_type(i.astype("intp"), "pd.Index[int]")
        assert_type(i.astype("p"), "pd.Index[int]")
        # pyarrow integer types
        assert_type(i.astype("int8[pyarrow]"), "pd.Index[int]")
        assert_type(i.astype("int16[pyarrow]"), "pd.Index[int]")
        assert_type(i.astype("int32[pyarrow]"), "pd.Index[int]")
        assert_type(i.astype("int64[pyarrow]"), "pd.Index[int]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_UINT_ARGS.items(), ids=repr)
def test_astype_uint(cast_arg: IntDtypeArg, target_type: type) -> None:
    i = pd.Index([1, 2, 3])
    check(i.astype(cast_arg), pd.Index, target_type)

    if TYPE_CHECKING:
        # pandas UInt8
        assert_type(i.astype(pd.UInt8Dtype()), "pd.Index[int]")
        assert_type(i.astype("UInt8"), "pd.Index[int]")
        # pandas UInt16
        assert_type(i.astype(pd.UInt16Dtype()), "pd.Index[int]")
        assert_type(i.astype("UInt16"), "pd.Index[int]")
        # pandas UInt32
        assert_type(i.astype(pd.UInt32Dtype()), "pd.Index[int]")
        assert_type(i.astype("UInt32"), "pd.Index[int]")
        # pandas UInt64
        assert_type(i.astype(pd.UInt64Dtype()), "pd.Index[int]")
        assert_type(i.astype("UInt64"), "pd.Index[int]")
        # numpy uint8
        assert_type(i.astype(np.ubyte), "pd.Index[int]")
        assert_type(i.astype("ubyte"), "pd.Index[int]")
        assert_type(i.astype("uint8"), "pd.Index[int]")
        assert_type(i.astype("B"), "pd.Index[int]")
        assert_type(i.astype("u1"), "pd.Index[int]")
        # numpy uint16
        assert_type(i.astype(np.ushort), "pd.Index[int]")
        assert_type(i.astype("ushort"), "pd.Index[int]")
        assert_type(i.astype("uint16"), "pd.Index[int]")
        assert_type(i.astype("H"), "pd.Index[int]")
        assert_type(i.astype("u2"), "pd.Index[int]")
        # numpy uint32
        assert_type(i.astype(np.uintc), "pd.Index[int]")
        assert_type(i.astype("uintc"), "pd.Index[int]")
        assert_type(i.astype("uint32"), "pd.Index[int]")
        assert_type(i.astype("I"), "pd.Index[int]")
        assert_type(i.astype("u4"), "pd.Index[int]")
        # numpy uint64
        assert_type(i.astype(np.uint), "pd.Index[int]")
        assert_type(i.astype("uint"), "pd.Index[int]")
        assert_type(i.astype("uint64"), "pd.Index[int]")
        assert_type(i.astype("ulong"), "pd.Index[int]")
        assert_type(i.astype("L"), "pd.Index[int]")
        assert_type(i.astype("u8"), "pd.Index[int]")
        # numpy unsigned pointer
        assert_type(i.astype(np.uintp), "pd.Index[int]")
        assert_type(i.astype("uintp"), "pd.Index[int]")
        assert_type(i.astype("P"), "pd.Index[int]")
        # pyarrow unsigned integer types
        assert_type(i.astype("uint8[pyarrow]"), "pd.Index[int]")
        assert_type(i.astype("uint16[pyarrow]"), "pd.Index[int]")
        assert_type(i.astype("uint32[pyarrow]"), "pd.Index[int]")
        assert_type(i.astype("uint64[pyarrow]"), "pd.Index[int]")
