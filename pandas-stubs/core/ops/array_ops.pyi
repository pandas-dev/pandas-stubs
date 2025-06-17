import numpy as np

from pandas.core.dtypes.generic import ABCExtensionArray as ABCExtensionArray

def arithmetic_op(left: np.ndarray | ABCExtensionArray, right, op, str_rep: str): ...
def comparison_op(
    left: np.ndarray | ABCExtensionArray, right, op
) -> np.ndarray | ABCExtensionArray: ...
