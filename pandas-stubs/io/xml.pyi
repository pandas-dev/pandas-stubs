from typing import Any

from pandas.core.frame import DataFrame

from pandas._libs.lib import NoDefaultDoNotUse
from pandas._typing import (
    CompressionOptions,
    ConvertersArg,
    DtypeArg,
    DtypeBackend,
    FilePath,
    ParseDatesArg,
    ReadBuffer,
    SequenceNotStr,
    StorageOptions,
    XMLParsers,
)

def read_xml(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    xpath: str = "./*",
    namespaces: dict[str, str] | None = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: SequenceNotStr[str] | None = None,
    dtype: DtypeArg | None = None,
    converters: ConvertersArg | None = None,
    parse_dates: ParseDatesArg[Any, Any] | None = None,
    # encoding can not be None for lxml and StringIO input
    encoding: str | None = "utf-8",
    parser: XMLParsers = "lxml",
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = None,
    iterparse: dict[str, list[str]] | None = None,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefaultDoNotUse = ...,
) -> DataFrame: ...
