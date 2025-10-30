from typing import TypeVar

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

class PrettyDict(dict[_KT, _VT]): ...
