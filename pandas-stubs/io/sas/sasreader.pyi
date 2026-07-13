from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Hashable,
    Iterator,
)
from types import TracebackType
from typing import (
    Literal,
    Self,
    overload,
)

from pandas import DataFrame

from pandas._typing import (
    CompressionOptions as CompressionOptions,
    FilePath as FilePath,
    ReadBuffer,
)

from pandas.io.sas.sas7bdat import SAS7BDATReader
from pandas.io.sas.sas_xport import XportReader

class SASReader(Iterator[DataFrame], ABC):
    @abstractmethod
    def read(self, nrows: int | None = None) -> DataFrame: ...
    @abstractmethod
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: Literal["sas7bdat"],
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> SAS7BDATReader: ...
@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: Literal["xport"],
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> XportReader: ...
@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> XportReader | SAS7BDATReader: ...
@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: Literal["sas7bdat"],
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: Literal[True],
    compression: CompressionOptions = "infer",
) -> SAS7BDATReader: ...
@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: Literal["xport"],
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: Literal[True],
    compression: CompressionOptions = "infer",
) -> XportReader: ...
@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: Literal[True],
    compression: CompressionOptions = "infer",
) -> XportReader | SAS7BDATReader: ...
@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: Literal["xport", "sas7bdat"] | None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: None = None,
    iterator: Literal[False] = False,
    compression: CompressionOptions = "infer",
) -> DataFrame: ...
