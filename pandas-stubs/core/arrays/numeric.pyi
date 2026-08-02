from pandas.core.arrays.masked import BaseMaskedArray
from typing_extensions import override

from pandas._libs.properties import cache_readonly

from pandas.core.dtypes.dtypes import BaseMaskedDtype

class NumericDtype(BaseMaskedDtype): ...

class NumericArray(BaseMaskedArray):
    @cache_readonly
    @override
    def dtype(self) -> NumericDtype: ...
