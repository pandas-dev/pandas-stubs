from typing import overload

import numpy as np

from pandas._typing import Scalar

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.dtypes import (
    register_extension_dtype as register_extension_dtype,
)

class SparseDtype(ExtensionDtype):
    @overload
    def __init__(
        self,
        dtype: type[bool | np.bool_],
        fill_value: bool | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: type[int | np.integer],
        fill_value: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: type[float | np.floating],
        fill_value: float | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: type[complex | np.complexfloating],
        fill_value: complex | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: type[np.datetime64],
        fill_value: np.datetime64 | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: type[np.timedelta64],
        fill_value: np.timedelta64 | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: type[str | bytes] | str | np.dtype[np.generic] | ExtensionDtype = ...,
        fill_value: Scalar | None = None,
    ) -> None: ...
    @property
    def subtype(
        self,
    ) -> (
        np.dtype
    ): ...  # TODO: pandas-dev/pandas-stubs#1654 make the class Generic so we can embed the subtype more precisely
    @property
    def fill_value(self) -> Scalar | None: ...
