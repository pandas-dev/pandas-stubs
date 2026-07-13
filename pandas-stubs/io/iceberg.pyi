from typing import Any

from pandas import DataFrame

def read_iceberg(
    table_identifier: str,
    catalog_name: str | None = None,
    *,
    catalog_properties: dict[str, Any] | None = None,
    columns: list[str] | None = None,
    row_filter: str | None = None,
    case_sensitive: bool = True,
    snapshot_id: int | None = None,
    limit: int | None = None,
    scan_properties: dict[str, Any] | None = None,
) -> DataFrame: ...
