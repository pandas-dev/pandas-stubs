from collections.abc import Callable
from typing import (
    Any,
    Literal,
    overload,
)

from pandas import (
    Index,
    Series,
)
from pandas.core.arrays.boolean import BooleanArray
from typing_extensions import Self

class NAType:
    def __new__(cls, *args: Any, **kwargs: Any) -> Self: ...
    def __format__(self, format_spec: str) -> str: ...
    def __hash__(self) -> int: ...
    def __reduce__(self) -> str: ...
    @overload
    def __add__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __add__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __add__(self, other: object, /) -> NAType: ...
    @overload
    def __radd__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __radd__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __radd__(self, other: object, /) -> NAType: ...
    @overload
    def __sub__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __sub__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __sub__(self, other: object, /) -> NAType: ...
    @overload
    def __rsub__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __rsub__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __rsub__(self, other: object, /) -> NAType: ...
    @overload
    def __mul__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __mul__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __mul__(self, other: object, /) -> NAType: ...
    @overload
    def __rmul__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __rmul__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __rmul__(self, other: object, /) -> NAType: ...
    def __matmul__(self, other: object, /) -> NAType: ...
    def __rmatmul__(self, other: object, /) -> NAType: ...
    @overload
    def __truediv__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __truediv__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __truediv__(self, other: object, /) -> NAType: ...
    @overload
    def __rtruediv__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __rtruediv__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __rtruediv__(self, other: object, /) -> NAType: ...
    @overload
    def __floordiv__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __floordiv__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __floordiv__(self, other: object, /) -> NAType: ...
    @overload
    def __rfloordiv__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __rfloordiv__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __rfloordiv__(self, other: object, /) -> NAType: ...
    @overload
    def __mod__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __mod__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __mod__(self, other: object, /) -> NAType: ...
    @overload
    def __rmod__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __rmod__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __rmod__(self, other: object, /) -> NAType: ...
    @overload
    def __divmod__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> tuple[Series, Series]: ...
    @overload
    def __divmod__(self, other: Index, /) -> tuple[Index, Index]: ...  # type: ignore[overload-overlap]
    @overload
    def __divmod__(self, other: object, /) -> tuple[NAType, NAType]: ...
    @overload
    def __rdivmod__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> tuple[Series, Series]: ...
    @overload
    def __rdivmod__(self, other: Index, /) -> tuple[Index, Index]: ...  # type: ignore[overload-overlap]
    @overload
    def __rdivmod__(self, other: object, /) -> tuple[NAType, NAType]: ...
    @overload  # type: ignore[override]
    def __eq__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series[bool]: ...
    @overload
    def __eq__(self, other: Index, /) -> BooleanArray: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: object, /
    ) -> NAType: ...
    @overload  # type: ignore[override]
    def __ne__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series[bool]: ...
    @overload
    def __ne__(self, other: Index, /) -> BooleanArray: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: object, /
    ) -> NAType: ...
    @overload
    def __le__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series[bool]: ...
    @overload
    def __le__(self, other: Index, /) -> BooleanArray: ...  # type: ignore[overload-overlap]
    @overload
    def __le__(self, other: object, /) -> NAType: ...
    @overload
    def __lt__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series[bool]: ...
    @overload
    def __lt__(self, other: Index, /) -> BooleanArray: ...  # type: ignore[overload-overlap]
    @overload
    def __lt__(self, other: object, /) -> NAType: ...
    @overload
    def __gt__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series[bool]: ...
    @overload
    def __gt__(self, other: Index, /) -> BooleanArray: ...  # type: ignore[overload-overlap]
    @overload
    def __gt__(self, other: object, /) -> NAType: ...
    @overload
    def __ge__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series[bool]: ...
    @overload
    def __ge__(self, other: Index, /) -> BooleanArray: ...  # type: ignore[overload-overlap]
    @overload
    def __ge__(self, other: object, /) -> NAType: ...
    def __neg__(self) -> NAType: ...
    def __pos__(self) -> NAType: ...
    def __abs__(self) -> NAType: ...
    def __invert__(self) -> NAType: ...
    @overload
    def __pow__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __pow__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __pow__(self, other: object, /) -> NAType: ...
    @overload
    def __rpow__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __rpow__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __rpow__(self, other: object, /) -> NAType: ...
    @overload
    def __and__(self, other: Literal[False], /) -> Literal[False]: ...  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
    @overload
    def __and__(self, other: bool | NAType, /) -> NAType: ...
    @overload
    def __rand__(self, other: Literal[False], /) -> Literal[False]: ...  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
    @overload
    def __rand__(self, other: bool, /) -> NAType: ...
    @overload
    def __or__(self, other: Literal[True], /) -> Literal[True]: ...  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
    @overload
    def __or__(self, other: bool | NAType, /) -> NAType: ...
    @overload
    def __ror__(self, other: Literal[True], /) -> Literal[True]: ...  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
    @overload
    def __ror__(self, other: bool | NAType, /) -> NAType: ...
    @overload
    def __xor__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __xor__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __xor__(self, other: object, /) -> NAType: ...
    @overload
    def __rxor__(  # type: ignore[overload-overlap] #  pyright: ignore[reportOverlappingOverload]
        self, other: Series, /
    ) -> Series: ...
    @overload
    def __rxor__(self, other: Index, /) -> Index: ...  # type: ignore[overload-overlap]
    @overload
    def __rxor__(self, other: object, /) -> NAType: ...
    __array_priority__: int
    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Any: ...

NA: NAType = ...
