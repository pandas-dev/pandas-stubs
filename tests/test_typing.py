from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np
import pandas as pd
import pytest

from pandas.core.dtypes.base import ExtensionDtype

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    get_dtype,
)
from tests.dtypes import DTYPE_ARG_ALIAS_MAPS

if TYPE_CHECKING:
    from pandas._typing import CovariantList


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
        assert actual == (
            type(expected) if isinstance(expected, ExtensionDtype) else expected
        )


@pytest.mark.parametrize(("dtype_arg", "alias_map"), DTYPE_ARG_ALIAS_MAPS.items())
def test_dtype_arg_aliases(dtype_arg: Any, alias_map: Mapping[Any, Any]) -> None:
    assert set(get_dtype(dtype_arg)) == {
        type(t) if isinstance(t, ExtensionDtype) else t for t in alias_map
    }


def test_covariant_list() -> None:
    def f(_: "CovariantList[float]") -> None: ...

    good1: list[float] = [1.0, 2.0, 3.0]  # OK, trivial case
    good2: list[int] = [1, 2, 3]  # OK, list[int] < list[float] due to covariance
    bad1: tuple[float, ...] = (1.0, 2.0, 3.0)  # Error, tuple is not a subtype of list
    bad2: list[str] = ["a", "b", "c"]  # Error, list[str] !< list[float]
    bad3: list[object] = [1, "a", 3.0]  # Error, list[object] !< list[float]
    bad4: float = 1.0  # Error, float !< list[float]

    f(good1)  # ty: ignore[invalid-argument-type]
    f(good2)  # ty: ignore[invalid-argument-type]
    if TYPE_CHECKING_INVALID_USAGE:
        f(bad1)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        f(bad2)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        f(bad3)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        f(bad4)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
