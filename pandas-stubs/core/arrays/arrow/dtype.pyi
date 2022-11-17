import sys
from typing import (
    Any,
    TypeAlias,
)

if sys.version_info < (3, 11):
    import pyarrow as pa
else:
    pa: TypeAlias = Any

from pandas.core.dtypes.base import StorageExtensionDtype

class ArrowDtype(StorageExtensionDtype):
    pyarrow_dtype: pa.DataType
    def __init__(self, pyarrow_dtype: pa.DataType) -> None: ...
