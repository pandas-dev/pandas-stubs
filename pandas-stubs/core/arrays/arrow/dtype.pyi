import sys
from typing import Any

import pyarrow as pa

from pandas.core.dtypes.base import StorageExtensionDtype

if sys.version_info < (3, 11):
    import pyarrow as pa
else:
    pa: Any

class ArrowDtype(StorageExtensionDtype):
    pyarrow_dtype: pa.DataType
    def __init__(self, pyarrow_dtype: pa.DataType) -> None: ...
