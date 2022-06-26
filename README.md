# pandas-stubs: Public type stubs for pandas

[![PyPI Latest Release](https://img.shields.io/pypi/v/pandas-stubs.svg)](https://pypi.org/project/pandas-stubs/)
[![Conda Latest Release](https://anaconda.org/conda-forge/pandas-stubs/badges/version.svg)](https://anaconda.org/conda-forge/pandas-stubs)
[![Package Status](https://img.shields.io/pypi/status/pandas-stubs.svg)](https://pypi.org/project/pandas-stubs/)
[![License](https://img.shields.io/pypi/l/pandas-stubs.svg)](https://github.com/pandas-dev/pandas-stubs/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/pandas-stubs?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/pandas-stubs)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pydata/pandas)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## What is it?

These are public type stubs for [**pandas**](http://pandas.pydata.org/), following the
convention of providing stubs in a separate package, as specified in [PEP 561](https://peps.python.org/pep-0561/#stub-only-packages).  The stubs cover the most typical use cases of
pandas.  In general, these stubs are narrower than what is possibly allowed by pandas,
but follow a convention of suggesting best recommended practices for using pandas.

The stubs are likely incomplete in terms of covering the published API of pandas.

The stubs are tested with [mypy](http://mypy-lang.org/) and [pyright](https://github.com/microsoft/pyright#readme) and are currently shipped with the Visual Studio Code extension
[pylance](https://github.com/microsoft/pylance-release#readme).

## Usage

Let’s take this example piece of code in file `round.py`

```python
import pandas as pd

decimals = pd.DataFrame({'TSLA': 2, 'AMZN': 1})
prices = pd.DataFrame(data={'date': ['2021-08-13', '2021-08-07', '2021-08-21'],
                            'TSLA': [720.13, 716.22, 731.22], 'AMZN': [3316.50, 3200.50, 3100.23]})
rounded_prices = prices.round(decimals=decimals)
```

Mypy won't see any issues with that, but after installing pandas-stubs and running it again:

```sh
mypy round.py
```

we get the following error message:

```text
round.py:6: error: Argument "decimals" to "round" of "DataFrame" has incompatible type "DataFrame"; expected "Union[int, Dict[Any, Any], Series[Any]]"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

And, if you use pyright:

```sh
pyright round.py
```

you get the following error message:

```text
 round.py:6:40 - error: Argument of type "DataFrame" cannot be assigned to parameter "decimals" of type "int | Dict[Unknown, Unknown] | Series[Unknown]" in function "round"
    Type "DataFrame" cannot be assigned to type "int | Dict[Unknown, Unknown] | Series[Unknown]"
      "DataFrame" is incompatible with "int"
      "DataFrame" is incompatible with "Dict[Unknown, Unknown]"
      "DataFrame" is incompatible with "Series[Unknown]" (reportGeneralTypeIssues)
```

And after confirming with the [docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.round.html)
we can fix the code:

```python
decimals = pd.Series({'TSLA': 2, 'AMZN': 1})
```

## Version Numbering Convention

The version number x.y.z.yymmdd corresponds to a test done with pandas version x.y.z, with the stubs released on the date mm/yy/dd.
It is anticipated that the stubs will be released more frequently than pandas as the stubs are expected to evolve due to more
public visibility.

## Where to get it

The source code is currently hosted on GitHub at: <https://github.com/pandas-dev/pandas-stubs>

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/pandas-stubs) and on [conda-forge](https://conda-forge.org/).

```sh
# conda
conda install pandas-stubs
```

```sh
# or PyPI
pip install pandas-stubs
```

## Dependencies

- [pandas: powerful Python data analysis toolkit](https://pandas.pydata.org/)
- [typing-extensions >= 4.2.0 - supporting the latest typing extensions](https://github.com/python/typing_extensions#readme)

## Installation from sources

- Make sure you have `python >= 3.8` installed.
- Install poetry

```sh
# conda
conda install poetry
```

```sh
# or PyPI
pip install poetry
```

- Install the project dependencies

```sh
poetry update -vvv
```

- Build and install the distribution

```sh
poetry run poe build_dist
poetry run poe install_dist
```

## License

[BSD 3](LICENSE)

## Documentation

Documentation is a work-in-progress.  

## Background

These stubs are the result of a strategic effort lead by the core pandas team to integrate [Microsoft type stub repository](https://github.com/microsoft/python-type-stubs)
together with the [VirtusLabs pandas_stubs repository](https://github.com/VirtusLab/pandas-stubs).

These stubs were initially forked from the Microsoft project <https://github.com/microsoft/python-type-stubs> as of [this commit](https://github.com/microsoft/python-type-stubs/tree/6b800063bde687cd1846122431e2a729a9de625a).

We are indebted to Microsoft and that project for the initial set of public type stubs.  We are also grateful for the original pandas-stubs project at <https://github.com/VirtusLab/pandas-stubs> that created the framework for testing the stubs.

## Getting help

Ask questions and report issues on the [pandas-stubs repository](https://github.com/pandas-dev/pandas-stubs/issues).  

## Discussion and Development

Most development discussions take place on GitHub in the [pandas-stubs repository](https://github.com/pandas-dev/pandas-stubs/). Further, the [pandas-dev mailing list](https://mail.python.org/mailman/listinfo/pandas-dev) can also be used for specialized discussions or design issues, and a [Gitter channel](https://gitter.im/pydata/pandas) is available for quick development related questions.

## Contributing to pandas-stubs

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.  See <https://github.com/pandas-dev/pandas-stubs/tree/main/docs/> for instructions.
