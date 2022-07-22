from typing import Any

from openpyxl.workbook import Workbook

from pandas._typing import (
    FilePath,
    ReadBuffer,
    Scalar,
    StorageOptions,
    WriteExcelBuffer,
)

from pandas.io.excel._base import (
    BaseExcelReader,
    ExcelWriter,
)

class OpenpyxlWriter(ExcelWriter):
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
    def book(self) -> Workbook: ...
    @property
    def sheets(self) -> dict[str, Any]: ...

class OpenpyxlReader(BaseExcelReader):
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer[bytes]): ...
    @property
    def sheet_names(self) -> list[str]: ...
    def get_sheet_by_name(self, name: str): ...
    def get_sheet_by_index(self, index: int): ...
    def get_sheet_data(
        self, sheet, convert_float: bool, file_rows_needed: int | None = ...
    ) -> list[list[Scalar]]: ...
