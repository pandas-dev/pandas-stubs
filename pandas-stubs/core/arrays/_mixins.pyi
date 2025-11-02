from pandas.core.arrays.base import ExtensionArray

from pandas._libs.arrays import NDArrayBacked

class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray): ...
