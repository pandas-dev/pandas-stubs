# from pandas import DataFrame, Int64Index, RangeIndex
from typing import Sequence

from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

def to_feather(df: DataFrame, path): ...
def read_feather(
    p: FilePathOrBuffer, columns: Sequence | None = ..., use_threads: bool = ...
): ...
