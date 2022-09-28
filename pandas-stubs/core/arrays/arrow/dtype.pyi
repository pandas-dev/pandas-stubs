import numpy as np
import pyarrow as pa

from pandas.core.dtypes.base import StorageExtensionDtype

class ArrowDtype(StorageExtensionDtype):
    pyarrow_dtype: pa.DataType
    def __init__(self, pyarrow_dtype: pa.DataType) -> None: ...
