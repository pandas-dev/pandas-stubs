from collections.abc import Mapping
from typing import (
    Any,
    Literal,
)

import numpy as np
import pandas as pd
import pytest

from tests import get_dtype
from tests.dtypes import DTYPE_ARG_ALIAS_MAPS


def test_get_dtype() -> None:
    alias_union = (
        Literal["int", "integer"] | np.long | pd.Int64Dtype | pd.DatetimeTZDtype
    )
    expected_values = [
        "int",
        "integer",
        np.long,
        pd.Int64Dtype(),
        pd.DatetimeTZDtype(tz="UTC"),
    ]
    for actual, expected in zip(get_dtype(alias_union), expected_values, strict=True):
        assert actual == expected


@pytest.mark.parametrize(("dtype_arg", "alias_map"), DTYPE_ARG_ALIAS_MAPS.items())
def test_dtype_arg_aliases(dtype_arg: Any, alias_map: Mapping[Any, Any]) -> None:
    assert set(get_dtype(dtype_arg)) == set(alias_map)
