import functools
import re

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import assert_type

from tests import check

# Separately define here so pytest works
np_ndarray_bool = npt.NDArray[np.bool_]


DATA = ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]


def test_string_accessors_type_preserving_series() -> None:
    s = pd.Series(DATA)
    _check = functools.partial(check, klass=pd.Series, dtype=str)
    _check(assert_type(s.str.capitalize(), "pd.Series[str]"))
    _check(assert_type(s.str.casefold(), "pd.Series[str]"))
    check(assert_type(s.str.cat(sep="X"), str), str)
    _check(assert_type(s.str.center(10), "pd.Series[str]"))
    _check(assert_type(s.str.get(2), "pd.Series[str]"))
    _check(assert_type(s.str.ljust(80), "pd.Series[str]"))
    _check(assert_type(s.str.lower(), "pd.Series[str]"))
    _check(assert_type(s.str.lstrip("a"), "pd.Series[str]"))
    _check(assert_type(s.str.normalize("NFD"), "pd.Series[str]"))
    _check(assert_type(s.str.pad(80, "right"), "pd.Series[str]"))
    _check(assert_type(s.str.removeprefix("a"), "pd.Series[str]"))
    _check(assert_type(s.str.removesuffix("e"), "pd.Series[str]"))
    _check(assert_type(s.str.repeat(2), "pd.Series[str]"))
    _check(assert_type(s.str.replace("a", "X"), "pd.Series[str]"))
    _check(assert_type(s.str.rjust(80), "pd.Series[str]"))
    _check(assert_type(s.str.rstrip(), "pd.Series[str]"))
    _check(assert_type(s.str.slice(0, 4, 2), "pd.Series[str]"))
    _check(assert_type(s.str.slice_replace(0, 2, "XX"), "pd.Series[str]"))
    _check(assert_type(s.str.strip(), "pd.Series[str]"))
    _check(assert_type(s.str.swapcase(), "pd.Series[str]"))
    _check(assert_type(s.str.title(), "pd.Series[str]"))
    _check(
        assert_type(s.str.translate({241: "n"}), "pd.Series[str]"),
    )
    _check(assert_type(s.str.upper(), "pd.Series[str]"))
    _check(assert_type(s.str.wrap(80), "pd.Series[str]"))
    _check(assert_type(s.str.zfill(10), "pd.Series[str]"))


def test_string_accessors_type_preserving_index() -> None:
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=pd.Index, dtype=str)
    _check(assert_type(idx.str.capitalize(), "pd.Index[str]"))
    _check(assert_type(idx.str.casefold(), "pd.Index[str]"))
    check(assert_type(idx.str.cat(sep="X"), str), str)
    _check(assert_type(idx.str.center(10), "pd.Index[str]"))
    _check(assert_type(idx.str.get(2), "pd.Index[str]"))
    _check(assert_type(idx.str.ljust(80), "pd.Index[str]"))
    _check(assert_type(idx.str.lower(), "pd.Index[str]"))
    _check(assert_type(idx.str.lstrip("a"), "pd.Index[str]"))
    _check(assert_type(idx.str.normalize("NFD"), "pd.Index[str]"))
    _check(assert_type(idx.str.pad(80, "right"), "pd.Index[str]"))
    _check(assert_type(idx.str.removeprefix("a"), "pd.Index[str]"))
    _check(assert_type(idx.str.removesuffix("e"), "pd.Index[str]"))
    _check(assert_type(idx.str.repeat(2), "pd.Index[str]"))
    _check(assert_type(idx.str.replace("a", "X"), "pd.Index[str]"))
    _check(assert_type(idx.str.rjust(80), "pd.Index[str]"))
    _check(assert_type(idx.str.rstrip(), "pd.Index[str]"))
    _check(assert_type(idx.str.slice(0, 4, 2), "pd.Index[str]"))
    _check(assert_type(idx.str.slice_replace(0, 2, "XX"), "pd.Index[str]"))
    _check(assert_type(idx.str.strip(), "pd.Index[str]"))
    _check(assert_type(idx.str.swapcase(), "pd.Index[str]"))
    _check(assert_type(idx.str.title(), "pd.Index[str]"))
    _check(
        assert_type(idx.str.translate({241: "n"}), "pd.Index[str]"),
    )
    _check(assert_type(idx.str.upper(), "pd.Index[str]"))
    _check(assert_type(idx.str.wrap(80), "pd.Index[str]"))
    _check(assert_type(idx.str.zfill(10), "pd.Index[str]"))


def test_string_accessors_boolean_series():
    s = pd.Series(DATA)
    _check = functools.partial(check, klass=pd.Series, dtype=bool)
    _check(assert_type(s.str.startswith("a"), "pd.Series[bool]"))
    _check(
        assert_type(s.str.startswith(("a", "b")), "pd.Series[bool]"),
    )
    _check(
        assert_type(s.str.contains("a"), "pd.Series[bool]"),
    )
    _check(
        assert_type(s.str.contains(re.compile(r"a")), "pd.Series[bool]"),
    )
    _check(assert_type(s.str.endswith("e"), "pd.Series[bool]"))
    _check(assert_type(s.str.endswith(("e", "f")), "pd.Series[bool]"))
    _check(assert_type(s.str.fullmatch("apple"), "pd.Series[bool]"))
    _check(assert_type(s.str.isalnum(), "pd.Series[bool]"))
    _check(assert_type(s.str.isalpha(), "pd.Series[bool]"))
    _check(assert_type(s.str.isdecimal(), "pd.Series[bool]"))
    _check(assert_type(s.str.isdigit(), "pd.Series[bool]"))
    _check(assert_type(s.str.isnumeric(), "pd.Series[bool]"))
    _check(assert_type(s.str.islower(), "pd.Series[bool]"))
    _check(assert_type(s.str.isspace(), "pd.Series[bool]"))
    _check(assert_type(s.str.istitle(), "pd.Series[bool]"))
    _check(assert_type(s.str.isupper(), "pd.Series[bool]"))
    _check(assert_type(s.str.match("pp"), "pd.Series[bool]"))


def test_string_accessors_boolean_index():
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=np.ndarray, dtype=np.bool_)
    _check(assert_type(idx.str.startswith("a"), np_ndarray_bool))
    _check(
        assert_type(idx.str.startswith(("a", "b")), np_ndarray_bool),
    )
    _check(
        assert_type(idx.str.contains("a"), np_ndarray_bool),
    )
    _check(
        assert_type(idx.str.contains(re.compile(r"a")), np_ndarray_bool),
    )
    _check(assert_type(idx.str.endswith("e"), np_ndarray_bool))
    _check(assert_type(idx.str.endswith(("e", "f")), np_ndarray_bool))
    _check(assert_type(idx.str.fullmatch("apple"), np_ndarray_bool))
    _check(assert_type(idx.str.isalnum(), np_ndarray_bool))
    _check(assert_type(idx.str.isalpha(), np_ndarray_bool))
    _check(assert_type(idx.str.isdecimal(), np_ndarray_bool))
    _check(assert_type(idx.str.isdigit(), np_ndarray_bool))
    _check(assert_type(idx.str.isnumeric(), np_ndarray_bool))
    _check(assert_type(idx.str.islower(), np_ndarray_bool))
    _check(assert_type(idx.str.isspace(), np_ndarray_bool))
    _check(assert_type(idx.str.istitle(), np_ndarray_bool))
    _check(assert_type(idx.str.isupper(), np_ndarray_bool))
    _check(assert_type(idx.str.match("pp"), np_ndarray_bool))


def test_string_accessors_integer_series():
    s = pd.Series(DATA)
    _check = functools.partial(check, klass=pd.Series, dtype=np.integer)
    _check(assert_type(s.str.find("p"), "pd.Series[int]"))
    _check(assert_type(s.str.index("p"), "pd.Series[int]"))
    _check(assert_type(s.str.rfind("e"), "pd.Series[int]"))
    _check(assert_type(s.str.rindex("p"), "pd.Series[int]"))
    _check(assert_type(s.str.count("pp"), "pd.Series[int]"))
    _check(assert_type(s.str.len(), "pd.Series[int]"))


def test_string_accessors_integer_index():
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=pd.Index, dtype=np.integer)
    _check(assert_type(idx.str.find("p"), "pd.Index[int]"))
    _check(assert_type(idx.str.index("p"), "pd.Index[int]"))
    _check(assert_type(idx.str.rfind("e"), "pd.Index[int]"))
    _check(assert_type(idx.str.rindex("p"), "pd.Index[int]"))
    _check(assert_type(idx.str.count("pp"), "pd.Index[int]"))
    _check(assert_type(idx.str.len(), "pd.Index[int]"))


def test_string_accessors_string_series():
    s = pd.Series([b"a1", b"b2", b"c3"])
    _check = functools.partial(check, klass=pd.Series, dtype=str)
    _check(assert_type(s.str.decode("utf-8"), "pd.Series[str]"))
    s2 = pd.Series([["apple", "banana"], ["cherry", "date"], [1, "eggplant"]])
    _check(assert_type(s2.str.join("-"), "pd.Series[str]"))


def test_string_accessors_string_index():
    idx = pd.Index([b"a1", b"b2", b"c3"])
    _check = functools.partial(check, klass=pd.Index, dtype=str)
    _check(assert_type(idx.str.decode("utf-8"), "pd.Index[str]"))
    idx2 = pd.Index([["apple", "banana"], ["cherry", "date"], [1, "eggplant"]])
    _check(assert_type(idx2.str.join("-"), "pd.Index[str]"))


def test_string_accessors_bytes_series():
    s = pd.Series(["a1", "b2", "c3"])
    check(assert_type(s.str.encode("latin-1"), "pd.Series[bytes]"), pd.Series, bytes)


def test_string_accessors_bytes_index():
    s = pd.Index(["a1", "b2", "c3"])
    check(assert_type(s.str.encode("latin-1"), "pd.Index[bytes]"), pd.Index, bytes)


def test_string_accessors_list():
    s = pd.Series(
        ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    )
    check(assert_type(s.str.findall("pp"), "pd.Series[list[str]]"), pd.Series, list)
    check(assert_type(s.str.split("a"), "pd.Series[list[str]]"), pd.Series, list)
    # GH 194
    check(
        assert_type(s.str.split("a", expand=False), "pd.Series[list[str]]"),
        pd.Series,
        list,
    )
    check(assert_type(s.str.rsplit("a"), "pd.Series[list[str]]"), pd.Series, list)
    check(
        assert_type(s.str.rsplit("a", expand=False), "pd.Series[list[str]]"),
        pd.Series,
        list,
    )


# def test_string_accessors_expanding():
#     check(assert_type(s3.str.extract(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s3.str.extractall(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.get_dummies(), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.partition("p"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.rpartition("p"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.rsplit("a", expand=True), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.split("a", expand=True), pd.DataFrame), pd.DataFrame)
