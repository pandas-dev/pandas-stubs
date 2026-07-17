from __future__ import annotations

import platform
from typing import (
    TYPE_CHECKING,
    assert_type,
)

import numpy as np
import pandas as pd
import pytest

from tests import check
from tests._typing import ComplexDtypeArg
from tests.dtypes import ASTYPE_COMPLEX_ARGS


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_COMPLEX_ARGS.items(), ids=repr)
def test_astype_complex(cast_arg: ComplexDtypeArg, target_type: type) -> None:
    i = pd.Index([1, 2, 3])

    if platform.system() == "Windows" and cast_arg in ("c32", "complex256"):
        with pytest.raises(TypeError):
            i.astype(cast_arg)
        pytest.skip("Windows does not support complex256")

    if (
        platform.system() == "Darwin"
        and platform.processor() == "arm"
        and cast_arg in ("c32", "complex256")
    ):
        with pytest.raises(TypeError):
            i.astype(cast_arg)
        pytest.skip("MacOS arm does not support complex256")

    check(i.astype(cast_arg), pd.Index, target_type)

    if TYPE_CHECKING:
        assert_type(i.astype(complex), "pd.Index[complex]")
        assert_type(i.astype("complex"), "pd.Index[complex]")
        # numpy complex64
        assert_type(i.astype(np.csingle), "pd.Index[complex]")
        assert_type(i.astype("csingle"), "pd.Index[complex]")
        assert_type(i.astype("complex64"), "pd.Index[complex]")
        assert_type(i.astype("F"), "pd.Index[complex]")
        assert_type(i.astype("c8"), "pd.Index[complex]")
        # numpy complex128
        assert_type(i.astype(np.cdouble), "pd.Index[complex]")
        assert_type(i.astype("cdouble"), "pd.Index[complex]")
        assert_type(i.astype("complex128"), "pd.Index[complex]")
        assert_type(i.astype("D"), "pd.Index[complex]")
        assert_type(i.astype("c16"), "pd.Index[complex]")
        # numpy complex256
        assert_type(i.astype(np.clongdouble), "pd.Index[complex]")
        assert_type(i.astype("clongdouble"), "pd.Index[complex]")
        assert_type(i.astype("complex256"), "pd.Index[complex]")
        assert_type(i.astype("G"), "pd.Index[complex]")
        assert_type(i.astype("c32"), "pd.Index[complex]")
