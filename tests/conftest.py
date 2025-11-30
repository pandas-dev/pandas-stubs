from collections.abc import Generator
from pathlib import Path
from shutil import copyfile

import pytest

# Use pandas._testing from the stubs for testing
groundtruth = Path(__file__).parents[1] / "pandas-stubs" / "_typing.pyi"
target = Path(__file__).parent / "_typing.py"
staging = Path(__file__).parent / "_typing.pyi"
copyfile(target, staging)
copyfile(groundtruth, target)


@pytest.fixture(autouse=True, scope="session")
def setup_typing_module() -> Generator[None, None, None]:
    """Ensure that tests._typing is recovered after running tests"""
    yield

    copyfile(staging, target)
    staging.unlink()
