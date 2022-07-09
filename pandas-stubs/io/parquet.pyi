from typing import (
    Optional,
    Sequence,
)

from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

def get_engine(engine: str) -> BaseImpl: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame): ...
    def write(self, df: DataFrame, path, compression, **kwargs): ...
    def read(self, path, columns=..., **kwargs) -> None: ...

class PyArrowImpl(BaseImpl):
    api = ...
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path,
        compression=...,
        coerce_timestamps=...,
        index: Optional[bool] = ...,
        partition_cols=...,
        **kwargs,
    ): ...
    def read(self, path, columns=..., **kwargs): ...

class FastParquetImpl(BaseImpl):
    api = ...
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path,
        compression=...,
        index=...,
        partition_cols=...,
        **kwargs,
    ): ...
    def read(self, path, columns=..., **kwargs): ...

def to_parquet(
    df: DataFrame,
    path,
    engine: str = ...,
    compression=...,
    index: Optional[bool] = ...,
    partition_cols=...,
    **kwargs,
): ...
def read_parquet(
    path: FilePathOrBuffer,
    engine: str = ...,
    columns: Optional[Sequence[str]] = ...,
    **kwargs,
) -> DataFrame: ...
