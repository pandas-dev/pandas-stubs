from typing import (
    List,
    Optional,
)

from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

def read_orc(
    path: FilePathOrBuffer, columns: Optional[List[str]] = ..., **kwargs
) -> DataFrame: ...
