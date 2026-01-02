from pandas.core.indexes.base import IndexSubclassBase

from pandas._typing import (
    S1,
    GenericT_co,
)

class ExtensionIndex(IndexSubclassBase[S1, GenericT_co]): ...
