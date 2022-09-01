from enum import Enum

no_default = None

class NoDefault(Enum):
    no_default: int

def infer_dtype(value: object, skipna: bool = ...) -> str: ...
