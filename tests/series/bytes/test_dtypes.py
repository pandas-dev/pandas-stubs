from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import BytesDtypeArg
from tests.dtypes import ASTYPE_BYTES_ARGS


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_BYTES_ARGS.items(), ids=repr)
def test_astype_bytes(cast_arg: BytesDtypeArg, target_type: type) -> None:
    s = pd.Series(["a", "b"])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python bytes
        assert_type(s.astype(bytes), "pd.Series[bytes]")
        assert_type(s.astype("bytes"), "pd.Series[bytes]")
        # numpy bytes
        assert_type(s.astype(np.bytes_), "pd.Series[bytes]")
        assert_type(s.astype("bytes_"), "pd.Series[bytes]")
        assert_type(s.astype("S"), "pd.Series[bytes]")
        # pyarrow bytes
        assert_type(s.astype("binary[pyarrow]"), "pd.Series[bytes]")
