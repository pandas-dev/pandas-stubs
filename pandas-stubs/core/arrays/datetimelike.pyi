from collections.abc import (
    Iterator,
    Sequence,
)
from typing import (
    Any,
    TypeAlias,
    overload,
)

from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from typing_extensions import Self

from pandas._libs import (
    NaT as NaT,
    NaTType as NaTType,
)
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    AxisInt,
    DatetimeLikeScalar,
    Frequency,
    NpDtype,
    PositionalIndexerTuple,
    ScalarIndexer,
    SequenceIndexer,
    TimeAmbiguous,
    TimeNonexistent,
    TimeUnit,
    np_1darray,
    np_1darray_str,
)

DTScalarOrNaT: TypeAlias = DatetimeLikeScalar | NaTType

class DatelikeOps:
    def strftime(self, date_format: str) -> np_1darray_str: ...

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

class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray):
    @property
    def ndim(self) -> int: ...
    def reshape(self, *args: Any, **kwargs: Any) -> Self: ...
    def __iter__(self) -> Iterator[Any]: ...
    @property
    def nbytes(self) -> int: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @property
    def size(self) -> int: ...
    @overload
    def __getitem__(  # pyrefly: ignore[bad-param-name-override]
        self, key: ScalarIndexer
    ) -> DTScalarOrNaT: ...
    @overload
    def __getitem__(  # ty: ignore[invalid-method-override]
        self, key: SequenceIndexer | PositionalIndexerTuple
    ) -> Self: ...
    def __setitem__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
        self, key: int | Sequence[int] | Sequence[bool] | slice, value: Any
    ) -> None: ...
    # TODO: pandas-dev/pandas-stubs#1589 import testing
    # def view(self, dtype: DtypeArg | None = None) -> np_ndarray: ...
    @property
    def freq(self) -> Frequency | None: ...
    @freq.setter
    def freq(self, value: Frequency) -> None: ...
    @property
    def freqstr(self) -> str | None: ...
    @property
    def inferred_freq(self) -> str | None: ...
    @property
    def resolution(self) -> str: ...
    # TODO: pandas-dev/pandas-stubs#1589 add testing for the below
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __iadd__(self, other: Any) -> Self: ...
    def __isub__(self, other: Any) -> Self: ...
    def min(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any
    ) -> Timestamp | Timedelta | NaTType: ...
    def max(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any
    ) -> Timestamp | Timedelta | NaTType: ...
    def mean(self, *, skipna: bool = True) -> Timestamp | Timedelta | NaTType: ...
