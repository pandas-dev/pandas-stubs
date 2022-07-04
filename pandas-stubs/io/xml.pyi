from typing import (
    Dict,
    Optional,
    Sequence,
    Union,
)

from pandas.core.frame import DataFrame as DataFrame

from pandas._typing import (
    CompressionOptions as CompressionOptions,
    ConvertersArg as ConvertersArg,
    DtypeArg as DtypeArg,
    FilePath as FilePath,
    ParseDatesArg as ParseDatesArg,
    ReadBuffer as ReadBuffer,
    StorageOptions as StorageOptions,
    XMLParsers as XMLParsers,
)

def read_xml(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    xpath: str = ...,
    namespaces: Optional[Dict[str, str]] = ...,
    elems_only: bool = ...,
    attrs_only: bool = ...,
    names: Optional[Sequence[str]] = ...,
    dtype: Optional[DtypeArg] = ...,
    converters: Optional[ConvertersArg] = ...,
    parse_dates: Optional[ParseDatesArg] = ...,
    # encoding can not be None for lxml and StringIO input
    encoding: Optional[str] = ...,
    parser: Optional[XMLParsers] = ...,
    stylesheet: Optional[Union[FilePath, ReadBuffer[bytes], ReadBuffer[str]]] = ...,
    iterparse: Optional[Dict[str, list[str]]] = ...,
    compression: Optional[CompressionOptions] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> DataFrame: ...
