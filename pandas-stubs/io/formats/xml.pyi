from typing import Any

from pandas.core.frame import DataFrame

from pandas._typing import (
    CompressionOptions,
    FilePath,
    ReadBuffer,
    StorageOptions,
    WriteBuffer,
)

class BaseXMLFormatter:
    frame = ...  # Incomplete
    path_or_buffer = ...  # Incomplete
    index = ...  # Incomplete
    root_name = ...  # Incomplete
    row_name = ...  # Incomplete
    na_rep = ...  # Incomplete
    attr_cols = ...  # Incomplete
    elem_cols = ...  # Incomplete
    namespaces = ...  # Incomplete
    prefix = ...  # Incomplete
    encoding = ...  # Incomplete
    xml_declaration = ...  # Incomplete
    pretty_print = ...  # Incomplete
    stylesheet = ...  # Incomplete
    compression = ...  # Incomplete
    storage_options = ...  # Incomplete
    orig_cols = ...  # Incomplete
    frame_dicts = ...  # Incomplete
    prefix_uri = ...  # Incomplete
    def __init__(
        self,
        frame: DataFrame,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = ...,
        index: bool = ...,
        root_name: str | None = ...,
        row_name: str | None = ...,
        na_rep: str | None = ...,
        attr_cols: list[str] | None = ...,
        elem_cols: list[str] | None = ...,
        namespaces: dict[str | None, str] | None = ...,
        prefix: str | None = ...,
        encoding: str = ...,
        xml_declaration: bool | None = ...,
        pretty_print: bool | None = ...,
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def build_tree(self) -> bytes: ...
    def validate_columns(self) -> None: ...
    def validate_encoding(self) -> None: ...
    def process_dataframe(self) -> dict[int | str, dict[str, Any]]: ...
    def handle_indexes(self) -> None: ...
    def get_prefix_uri(self) -> str: ...
    def other_namespaces(self) -> dict: ...
    def build_attribs(self, d: dict[str, Any], elem_row: Any) -> Any: ...
    def build_elems(self, d: dict[str, Any], elem_row: Any) -> None: ...
    def write_output(self) -> str | None: ...

class EtreeXMLFormatter(BaseXMLFormatter):
    root = ...  # Incomplete
    elem_cols = ...  # Incomplete
    out_xml = ...  # Incomplete
    def build_tree(self) -> bytes: ...
    def get_prefix_uri(self) -> str: ...
    def build_elems(self, d: dict[str, Any], elem_row: Any) -> None: ...
    def prettify_tree(self) -> bytes: ...
    def add_declaration(self) -> bytes: ...
    def remove_declaration(self) -> bytes: ...

class LxmlXMLFormatter(BaseXMLFormatter):
    def __init__(self, *args, **kwargs) -> None: ...
    root = ...  # Incomplete
    elem_cols = ...  # Incomplete
    out_xml = ...  # Incomplete
    def build_tree(self) -> bytes: ...
    def convert_empty_str_key(self) -> None: ...
    def get_prefix_uri(self) -> str: ...
    def build_elems(self, d: dict[str, Any], elem_row: Any) -> None: ...
    def transform_doc(self) -> bytes: ...
