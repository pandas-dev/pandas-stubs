from typing import (
    Any,
    Never,
    Self,
)

import numpy as np
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import override

from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    S1,
    AxisIndex,
    GenericT_co,
    TimeUnit,
    np_ndarray_complex,
)

class DatetimeIndexOpsMixin(ExtensionIndex[S1, GenericT_co]):
    @property
    def freq(self) -> BaseOffset | None: ...
    @property
    def freqstr(self) -> str | None: ...
    @property
    def is_all_dates(self) -> bool: ...
    @override
    def min(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> S1: ...
    @override
    def argmin(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    @override
    def max(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> S1: ...
    @override
    def argmax(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    @override
    def __mul__(self, other: np_ndarray_complex, /) -> Never: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
    @override
    def __rmul__(self, other: np_ndarray_complex, /) -> Never: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin[S1, GenericT_co]):
    @property
    def unit(self) -> TimeUnit: ...
    def as_unit(self, unit: TimeUnit) -> Self: ...
