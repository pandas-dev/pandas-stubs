from __future__ import annotations

import platform
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import ComplexDtypeArg
from tests.dtypes import ASTYPE_COMPLEX_ARGS


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_COMPLEX_ARGS.items(), ids=repr)
def test_astype_complex(cast_arg: ComplexDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])

    if platform.system() == "Windows" and cast_arg in ("c32", "complex256"):
        with pytest.raises(TypeError):
            s.astype(cast_arg)
        pytest.skip("Windows does not support complex256")

    if (
        platform.system() == "Darwin"
        and platform.processor() == "arm"
        and cast_arg in ("c32", "complex256")
    ):
        with pytest.raises(TypeError):
            s.astype(cast_arg)
        pytest.skip("MacOS arm does not support complex256")

    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        assert_type(s.astype(complex), "pd.Series[complex]")
        assert_type(s.astype("complex"), "pd.Series[complex]")
        # numpy complex64
        assert_type(s.astype(np.csingle), "pd.Series[complex]")
        assert_type(s.astype("csingle"), "pd.Series[complex]")
        assert_type(s.astype("complex64"), "pd.Series[complex]")
        assert_type(s.astype("F"), "pd.Series[complex]")
        assert_type(s.astype("c8"), "pd.Series[complex]")
        # numpy complex128
        assert_type(s.astype(np.cdouble), "pd.Series[complex]")
        assert_type(s.astype("cdouble"), "pd.Series[complex]")
        assert_type(s.astype("complex128"), "pd.Series[complex]")
        assert_type(s.astype("D"), "pd.Series[complex]")
        assert_type(s.astype("c16"), "pd.Series[complex]")
        # numpy complex256
        assert_type(s.astype(np.clongdouble), "pd.Series[complex]")
        assert_type(s.astype("clongdouble"), "pd.Series[complex]")
        assert_type(s.astype("complex256"), "pd.Series[complex]")
        assert_type(s.astype("G"), "pd.Series[complex]")
        assert_type(s.astype("c32"), "pd.Series[complex]")
