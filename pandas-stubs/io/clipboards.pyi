from __future__ import annotations

from typing import Any

def read_clipboard(sep: str = ..., **kwargs): ...
def to_clipboard(
    obj, excel: bool | None = ..., sep: str | None = ..., **kwargs: Any
) -> None: ...
