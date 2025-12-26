from collections.abc import Mapping
from typing import Any

import pytest

from tests import (
    get_dtype,
    get_dtype_arg_alias_maps,
)


@pytest.mark.parametrize(("dtype_arg", "alias_map"), get_dtype_arg_alias_maps().items())
def test_dtype_arg_aliases(dtype_arg: Any, alias_map: Mapping[Any, Any]) -> None:
    assert set(get_dtype(dtype_arg)) == set(alias_map)
