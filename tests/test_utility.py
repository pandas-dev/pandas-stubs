import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check


def test_show_version():
    with pytest.warns(UserWarning, match="Setuptools is replacing distutils"):
        check(assert_type(pd.show_versions(True), None), type(None))
        check(assert_type(pd.show_versions(False), None), type(None))
