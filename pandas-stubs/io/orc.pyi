from typing import (
    List,
    Optional,
)

from pandas.core.frame import DataFrame as DataFrame

from pandas._typing import FilePathOrBuffer as FilePathOrBuffer

def read_orc(
    path: FilePathOrBuffer, columns: Optional[List[str]] = ..., **kwargs
) -> DataFrame: ...
