from typing import (
    Any,
    Generic,
    Literal,
    TypeAlias,
    overload,
    type_check_only,
)

from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pyarrow as pa
from typing_extensions import (
    Self,
    TypeVar,
)

from pandas._libs.missing import NAType
from pandas._typing import (
    DtypeArg,
    np_ndarray_object,
)

from pandas.core.dtypes.base import ExtensionDtype

Storage: TypeAlias = Literal["python", "pyarrow"]
StorageT = TypeVar("StorageT", bound=Storage)
_StorageT = TypeVar("_StorageT", bound=Storage | None, default=None)

# Trick to make mypy happy
@type_check_only
class _StringDtypeStorageDescriptor:
    @overload
    def __get__(
        self, instance: StringDtype[None], owner: type[StringDtype[None]]
    ) -> Storage: ...
    @overload
    def __get__(
        self, instance: StringDtype[StorageT], owner: type[StringDtype[StorageT]]
    ) -> StorageT: ...

class StringDtype(ExtensionDtype, Generic[_StorageT]):
    @overload
    def __new__(
        cls, storage: StorageT, na_value: NAType | float = ...
    ) -> StringDtype[StorageT]: ...
    @overload
    def __new__(
        cls, storage: None = None, na_value: NAType | float = ...
    ) -> StringDtype: ...
    storage = _StringDtypeStorageDescriptor()
    @property
    def na_value(self) -> NAType | float: ...

class BaseStringArray(ExtensionArray, Generic[_StorageT]):
    @property
    def dtype(self) -> StringDtype[_StorageT]: ...

class StringArray(BaseStringArray[Literal["python"]], NumpyExtensionArray):
    def __init__(
        self, values: np_ndarray_object | Self, copy: bool = False
    ) -> None: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> pa.StringArray: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
