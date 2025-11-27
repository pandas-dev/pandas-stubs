from collections.abc import Generator
from pathlib import Path
from shutil import copyfile

import pytest

target = Path(__file__).parent / "_typing.py"
copyfile(Path(__file__).parents[1] / "pandas-stubs" / "_typing.pyi", target)


@pytest.fixture(autouse=True, scope="session")
def setup_typing_module() -> Generator[None, None, None]:
    """Ensure that tests._typing is removed after running tests"""
    yield
    target.unlink()
