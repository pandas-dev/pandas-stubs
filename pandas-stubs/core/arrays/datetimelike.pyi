from collections.abc import Sequence
from datetime import datetime
from typing import (
    TypeAlias,
    overload,
)

import numpy as np
import numpy.typing as npt
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
)
from typing_extensions import Self

from pandas._libs import (
    NaT as NaT,
    NaTType as NaTType,
)
from pandas._typing import (
    DatetimeLikeScalar,
    DtypeArg,
    Frequency,
    NpDtype,
    PositionalIndexerTuple,
    ScalarIndexer,
    SequenceIndexer,
    TimeAmbiguous,
    TimeNonexistent,
    TimeUnit,
    np_1darray,
)

DTScalarOrNaT: TypeAlias = DatetimeLikeScalar | NaTType

class DatelikeOps:
    def strftime(self, date_format: str) -> npt.NDArray[np.object_]: ...

class TimelikeOps:
    @property
    def unit(self) -> TimeUnit: ...
    def as_unit(self, unit: TimeUnit) -> Self: ...
    def round(
        self,
        freq: Frequency,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self: ...
    def floor(
        self,
        freq: Frequency,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self: ...
    def ceil(
        self,
        freq: Frequency,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self: ...

class DatetimeLikeArrayMixin(ExtensionOpsMixin, ExtensionArray):
    @property
    def ndim(self) -> int: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @property
    def size(self) -> int: ...
    @overload
    def __getitem__(  # pyrefly: ignore[bad-override,bad-param-name-override]
        self, key: ScalarIndexer
    ) -> DTScalarOrNaT: ...
    @overload
    def __getitem__(  # ty: ignore[invalid-method-override]
        self, key: SequenceIndexer | PositionalIndexerTuple
    ) -> Self: ...
    def __setitem__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: int | Sequence[int] | Sequence[bool] | slice, value: datetime
    ) -> None: ...
    def view(self, dtype: DtypeArg | None = None) -> Self: ...
    def copy(self) -> Self: ...
    @property
    def freq(self) -> Frequency | None: ...
    @freq.setter
    def freq(self, value: Frequency | None) -> None: ...
    @property
    def freqstr(self) -> str | None: ...
    @property
    def inferred_freq(self) -> str | None: ...
    @property
    def resolution(self) -> str: ...
    __pow__ = ...
    __rpow__ = ...
    __rmul__ = ...
