from typing import Any

from pandas._typing import (
    FilePath,
    StorageOptions,
    WriteExcelBuffer,
)

from pandas.io.excel._base import ExcelWriter

class XlwtWriter(ExcelWriter):
    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = ...,
        date_format: str | None = ...,
        datetime_format: str | None = ...,
        encoding: str | None = ...,
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
    @property
    def fm_date(self): ...
    @property
    def fm_datetime(self): ...
