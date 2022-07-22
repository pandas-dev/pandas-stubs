from typing import Any

from _typeshed import Incomplete

from pandas._typing import (
    FilePath,
    StorageOptions,
    WriteExcelBuffer,
)

from pandas.io.excel._base import ExcelWriter

class _XlsxStyler:
    STYLE_MAPPING: dict[str, list[tuple[tuple[str, ...], str]]]
    @classmethod
    def convert(cls, style_dict, num_format_str: Incomplete | None = ...): ...

class XlsxWriter(ExcelWriter):
    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = ...,
        date_format: str | None = ...,
        datetime_format: str | None = ...,
        mode: str = ...,
        storage_options: StorageOptions = ...,
        if_sheet_exists: str | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def book(self): ...
    @property
    def sheets(self) -> dict[str, Any]: ...
