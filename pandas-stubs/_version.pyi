from typing import (
    Literal,
    TypedDict,
)

_VersionInfo = TypedDict(
    "_VersionInfo",
    {
        "date": str,
        "dirty": bool,
        "error": str | None,
        "full-revisionid": str,
        "version": str,
    },
)

version_json: str = ...

def get_versions() -> _VersionInfo: ...

_stub_version: Literal["1.5.1.221024"]
