from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray): ...
