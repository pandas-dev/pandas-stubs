from datetime import timezone
from typing import (
    Any,
    Generic,
    Literal,
    Never,
    Self,
    TypeAlias,
    overload,
)

import numpy as np
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import TypeVar

from pandas._libs import NaTType
from pandas._libs.missing import NAType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.offsets import (
    RelativeDeltaOffset,
    SingleConstructorOffset,
)
from pandas._typing import (
    NumpyTimedeltaDtypeArg,
    NumpyTimestampDtypeArg,
    Ordered,
    Scalar,
    TimeZones,
    npt,
)

from pandas.core.dtypes.base import (
    ExtensionDtype as ExtensionDtype,
    register_extension_dtype as register_extension_dtype,
)

_dt_units: TypeAlias = Literal["s", "ms", "us", "ns"]

CategoricalValueT = TypeVar(
    "CategoricalValueT", str, int, float, object, default=object
)
CategoricalValueT1 = TypeVar("CategoricalValueT1", str, int, float)

class BaseMaskedDtype(ExtensionDtype):
    @property
    def na_value(self) -> NAType: ...

class PandasExtensionDtype(ExtensionDtype): ...

class CategoricalDtype(
    PandasExtensionDtype, ExtensionDtype, Generic[CategoricalValueT]
):
    def __init__(
        self,
        categories: Series | Index | list[Any] | None = ...,
        ordered: Ordered = ...,
    ) -> None: ...
    @property
    def categories(self) -> Index: ...
    @property
    def ordered(self) -> Ordered: ...

@register_extension_dtype
class DatetimeTZDtype(PandasExtensionDtype):
    @overload
    def __init__(self, unit: _dt_units | Self = "ns", *, tz: TimeZones) -> None: ...
    @overload
    def __init__(self, unit: _dt_units | Self, tz: TimeZones) -> None: ...
    @property
    def unit(self) -> _dt_units: ...
    @property
    def tz(self) -> timezone: ...
    @property
    def na_value(self) -> NaTType: ...

class PeriodDtype(PandasExtensionDtype):
    def __init__(
        self, freq: str | SingleConstructorOffset | RelativeDeltaOffset = ...
    ) -> None: ...
    @property
    def freq(self) -> BaseOffset: ...
    @property
    def na_value(self) -> NaTType: ...

class IntervalDtype(PandasExtensionDtype):
    def __init__(self, subtype: str | npt.DTypeLike | None = ...) -> None: ...
    @property
    def subtype(self) -> np.dtype | None: ...

class SparseDtype(ExtensionDtype):
    @overload
    def __new__(
        cls, dtype: type[bool | np.bool_], fill_value: bool | None = None
    ) -> Self: ...
    @overload
    def __new__(
        cls, dtype: type[int | np.integer], fill_value: int | None = None
    ) -> Self: ...
    @overload
    def __new__(
        cls, dtype: type[float | np.floating], fill_value: float | None = None
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        dtype: type[complex | np.complexfloating],
        fill_value: complex | None = None,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        dtype: type[np.datetime64 | np.timedelta64],
        fill_value: np.datetime64 | None = None,
    ) -> Never: ...
    @overload
    def __new__(
        cls, dtype: NumpyTimestampDtypeArg, fill_value: np.datetime64 | None = None
    ) -> Self: ...
    @overload
    def __new__(
        cls, dtype: NumpyTimedeltaDtypeArg, fill_value: np.timedelta64 | None = None
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        dtype: type[str | bytes] | str | np.dtype[np.generic] | ExtensionDtype = ...,
        fill_value: Scalar | None = None,
    ) -> Self: ...
    # TODO: pandas-dev/pandas-stubs#1654 make the class Generic so we can embed the subtype more precisely
    @property
    def subtype(self) -> np.dtype: ...
    @property
    def fill_value(self) -> Scalar | None: ...
