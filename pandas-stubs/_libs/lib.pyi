from __future__ import annotations

from enum import Enum

no_default = None

from typing import Literal

class _NoDefault(Enum):
    no_default = ...

NoDefault = Literal[_NoDefault.no_default]

def infer_dtype(value: object, skipna: bool = ...) -> str: ...
