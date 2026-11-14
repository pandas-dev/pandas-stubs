from typing import (
    Generic,
    Literal,
    Self,
    TypeAlias,
    overload,
    type_check_only,
)

from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pyarrow as pa
from typing_extensions import (
    TypeVar,
    override,
)

from pandas._libs.missing import NAType
from pandas._typing import (
    DtypeArg,
    ScalarIndexer,
    SequenceIndexer,
    np_ndarray_object,
)

from pandas.core.dtypes.base import ExtensionDtype

Storage: TypeAlias = Literal["python", "pyarrow"]
StorageT = TypeVar("StorageT", bound=Storage)
_StorageT = TypeVar("_StorageT", bound=Storage | None, covariant=True, default=None)
_NaValueT = TypeVar("_NaValueT", bound=NAType | float, covariant=True, default=NAType)
_StringArrayNaValueT = TypeVar(
    "_StringArrayNaValueT", bound=NAType | float, covariant=True, default=NAType
)
_BaseStringNaValueT = TypeVar(
    "_BaseStringNaValueT",
    bound=NAType | float,
    covariant=True,
    default=NAType,
)

# Trick to make mypy happy
@type_check_only
class _StringDtypeStorageDescriptor:
    @overload
    def __get__(
        self,
        instance: StringDtype[None, _NaValueT],
        owner: type[StringDtype[None, _NaValueT]],
        /,
    ) -> Storage: ...
    @overload
    def __get__(
        self,
        instance: StringDtype[StorageT, _NaValueT],
        owner: type[StringDtype[StorageT, _NaValueT]],
        /,
    ) -> StorageT: ...
    @overload
    def __get__(
        self,
        instance: StringDtype[Storage | None, _NaValueT],
        owner: type[StringDtype[Storage | None, _NaValueT]],
        /,
    ) -> Storage: ...

class StringDtype(ExtensionDtype, Generic[_StorageT, _NaValueT]):
    @overload
    def __new__(
        cls, storage: StorageT, na_value: _NaValueT = ...
    ) -> StringDtype[StorageT, _NaValueT]: ...
    @overload
    def __new__(
        cls, storage: None = None, na_value: _NaValueT = ...
    ) -> StringDtype[None, _NaValueT]: ...
    storage = _StringDtypeStorageDescriptor()
    @property
    @override
    def na_value(self) -> _NaValueT: ...

class BaseStringArray(ExtensionArray, Generic[_StorageT, _BaseStringNaValueT]):
    @overload
    @override
    def __getitem__(self, item: ScalarIndexer, /) -> str | _BaseStringNaValueT: ...
    @overload
    def __getitem__(self, item: SequenceIndexer, /) -> Self: ...
    @override
    def item(self, index: int | None = None) -> str | _BaseStringNaValueT: ...
    @property
    @override
    def dtype(self) -> StringDtype[_StorageT, _BaseStringNaValueT]: ...

class StringArray(
    BaseStringArray[Literal["python"], _StringArrayNaValueT],
    NumpyExtensionArray,
    Generic[_StringArrayNaValueT],
):
    def __init__(
        self, values: np_ndarray_object | Self, copy: bool = False
    ) -> None: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> pa.StringArray: ...
