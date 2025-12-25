from typing import (
    Any,
    Literal,
)

from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pyarrow as pa
from typing_extensions import Self

from pandas._libs.missing import NAType
from pandas._typing import (
    DtypeArg,
    np_ndarray_object,
    np_ndarray_str,
)

from pandas.core.dtypes.base import ExtensionDtype

class StringDtype(ExtensionDtype):
    def __new__(
        cls,
        storage: Literal["python", "pyarrow"] | None = None,
        na_value: NAType | float = ...,
    ) -> Self: ...
    @property
    def storage(self) -> Literal["python", "pyarrow"]: ...
    @property
    def na_value(self) -> NAType | float: ...

class BaseStringArray(ExtensionArray):
    @property
    def dtype(self) -> StringDtype: ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    def __new__(
        cls, values: np_ndarray_object | np_ndarray_str, copy: bool = False
    ) -> Self: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> pa.StringArray: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
