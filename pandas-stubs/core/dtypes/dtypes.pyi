from datetime import timezone
import sys
from typing import (
    Any,
    Literal,
    TypeAlias,
    overload,
)

import numpy as np
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs import NaTType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.offsets import (
    RelativeDeltaOffset,
    SingleConstructorOffset,
)
from pandas._typing import (
    Ordered,
    TimeZones,
    npt,
)

from pandas.core.dtypes.base import (
    ExtensionDtype as ExtensionDtype,
    register_extension_dtype as register_extension_dtype,
)

_dt_units: TypeAlias = Literal["s", "ms", "us", "ns"]

class BaseMaskedDtype(ExtensionDtype): ...
class PandasExtensionDtype(ExtensionDtype): ...

class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
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
    if sys.version_info >= (3, 11):
        @property
        def subtype(self) -> np.dtype | None: ...
    else:
        @property
        def subtype(self) -> np.dtype[Any] | None: ...

class SparseDtype(ExtensionDtype): ...
