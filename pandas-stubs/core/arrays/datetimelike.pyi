from collections.abc import (
    Iterator,
    Sequence,
)
from typing import (
    Any,
    TypeAlias,
    overload,
)

from pandas.core.arrays.base import ExtensionArray
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs import (
    NaT as NaT,
    NaTType as NaTType,
)
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    AnyArrayLikeInt,
    AxisInt,
    DatetimeLikeScalar,
    DtypeArg,
    Frequency,
    NpDtype,
    PositionalIndexerTuple,
    Renamer,
    ScalarIndexer,
    SequenceIndexer,
    TimeAmbiguous,
    TimeNonexistent,
    TimeUnit,
    np_1darray,
    np_1darray_str,
    np_ndarray_bool,
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

class DatetimeLikeArrayMixin(ExtensionArray):
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int]: ...
    def reshape(self, *args: Any, **kwargs: Any) -> Self: ...
    def ravel(self, *args: Any, **kwargs: Any) -> Self: ...
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
    def __setitem__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
        self, key: int | Sequence[int] | Sequence[bool] | slice, value: Any
    ) -> None: ...
    def view(self, dtype: DtypeArg | None = None) -> Self: ...
    def unique(self) -> Self: ...
    def copy(self) -> Self: ...
    def shift(self, periods: int = 1, fill_value: object | None = None) -> Self: ...
    def repeat(
        self,
        repeats: int | AnyArrayLikeInt | Sequence[int],
        axis: AxisInt | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    def value_counts(
        self, dropna: bool = True
    ) -> Series[int]: ...  # probably to put in base class
    def map(self, mapper: Renamer) -> Self: ...
    def isna(self) -> np_ndarray_bool | ExtensionArray: ...
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
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __iadd__(self, other: Any) -> Self: ...
    def __isub__(self, other: Any) -> Self: ...
    def min(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any
    ) -> Timestamp | Timedelta: ...
    def max(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any
    ) -> Timestamp | Timedelta: ...
    def mean(self, *, skipna: bool = True) -> Timestamp | Timedelta | NaTType: ...
