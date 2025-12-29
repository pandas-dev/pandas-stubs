from typing import (
    Any,
    Generic,
    Literal,
)

from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from typing_extensions import (
    Self,
    TypeVar,
)

from pandas._libs.missing import NAType
from pandas._typing import (
    AnyArrayLike,
    DtypeArg,
)

from pandas.core.dtypes.base import ExtensionDtype

StorageT = TypeVar(
    "StorageT", bound=Literal["python", "pyarrow"], default=Literal["python", "pyarrow"]
)

class StringDtype(ExtensionDtype, Generic[StorageT]):
    def __new__(
        cls, storage: StorageT | None = None, na_value: NAType | float = ...
    ) -> Self: ...
    @property
    def storage(self) -> StorageT: ...
    @property
    def na_value(self) -> NAType | float: ...

class BaseStringArray(ExtensionArray, Generic[StorageT]):
    @property
    def dtype(self) -> StringDtype[StorageT]: ...

class StringArray(BaseStringArray[Literal["python"]], NumpyExtensionArray):
    def __init__(self, values: AnyArrayLike, copy: bool = False) -> None: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
