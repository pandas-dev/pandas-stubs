from __future__ import annotations

from typing import Any

def parse_time_string(*args, **kwargs) -> Any: ...

class DateParseError(ValueError):
    def __init__(self, *args, **kwargs) -> None: ...
