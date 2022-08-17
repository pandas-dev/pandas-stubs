from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import (
    IO,
    Any,
)
import uuid

import pandas as pd
from pandas import DataFrame
from typing_extensions import assert_type

from tests import check

from pandas.io.stata import (
    StataReader,
    read_stata,
)

DF = DataFrame({"a": [1, 2, 3], "b": [0.0, 0.0, 0.0]})


@contextmanager
def ensure_clean(filename=None, return_filelike: bool = False, **kwargs: Any):
    """
    Gets a temporary path and agrees to remove on close.
    This implementation does not use tempfile.mkstemp to avoid having a file handle.
    If the code using the returned path wants to delete the file itself, windows
    requires that no program has a file handle to it.
    Parameters
    ----------
    filename : str (optional)
        suffix of the created file.
    return_filelike : bool (default False)
        if True, returns a file-like which is *always* cleaned. Necessary for
        savefig and other functions which want to append extensions.
    **kwargs
        Additional keywords are passed to open().
    """
    folder = Path(tempfile.gettempdir())

    if filename is None:
        filename = ""
    filename = str(uuid.uuid4()) + filename
    path = folder / filename

    path.touch()

    handle_or_str: str | IO = str(path)
    if return_filelike:
        kwargs.setdefault("mode", "w+b")
        handle_or_str = open(path, **kwargs)

    try:
        yield handle_or_str
    finally:
        if not isinstance(handle_or_str, str):
            handle_or_str.close()
        if path.is_file():
            path.unlink()


def test_read_stata_df():
    with ensure_clean() as path:
        DF.to_stata(path)
        check(assert_type(read_stata(path), pd.DataFrame), pd.DataFrame)


def test_read_stata_iterator_positional():
    with ensure_clean() as path:
        str_path = str(path)
        DF.to_stata(str_path)
        check(
            assert_type(
                read_stata(
                    str_path, False, False, None, False, False, None, False, 2, True
                ),
                StataReader,
            ),
            StataReader,
        )


def test_read_stata_iterator():
    with ensure_clean() as path:
        str_path = str(path)
        DF.to_stata(str_path)
        check(
            assert_type(read_stata(str_path, iterator=True), StataReader), StataReader
        )
