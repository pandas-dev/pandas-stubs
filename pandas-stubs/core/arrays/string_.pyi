from typing import (
    Any,
    Generic,
    Literal,
    overload,
)

from pandas.core.arrays import PandasArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.series import Series
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
    @overload
    def __new__(
        cls,
        storage: Literal["python"],
        na_value: NAType | float = ...,
    ) -> StringDtype[Literal["python"]]: ...
    @overload
    def __new__(
        cls,
        storage: Literal["pyarrow"],
        na_value: NAType | float = ...,
    ) -> StringDtype[Literal["pyarrow"]]: ...
    @overload
    def __new__(cls, storage: None = None, na_value: NAType | float = ...) -> Self: ...
    @property
    def storage(self) -> StorageT: ...
    @property
    def na_value(self) -> NAType | float: ...

class BaseStringArray(ExtensionArray):
    @property
    def dtype(self) -> StringDtype: ...

class StringArray(BaseStringArray, PandasArray):
    def __init__(self, values: AnyArrayLike, copy: bool = False) -> None: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def value_counts(self, dropna: bool = True) -> Series[int]: ...
