import pyarrow as pa
from typing_extensions import override

from pandas._libs.missing import NAType

from pandas.core.dtypes.base import StorageExtensionDtype

class ArrowDtype(StorageExtensionDtype):
    pyarrow_dtype: pa.DataType
    def __init__(self, pyarrow_dtype: pa.DataType) -> None: ...
    @property
    @override
    def na_value(self) -> NAType: ...
