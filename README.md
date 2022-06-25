 <div>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/BrenoJesusFernandes/brenojesusfernandes/blob/main/img/pandas.svg">
    <img src="https://pandas.pydata.org/static/img/pandas.svg">
  </picture> <br>
</div>

-----------------

# pandas-stubs: type annotations for pandas
[![CI/CD](https://github.com/BrenoJesusFernandes/pandas-stubs/actions/workflows/pipeline.yml/badge.svg)](https://github.com/BrenoJesusFernandes/pandas-stubs/actions/workflows/pipeline.yml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/pandas-stubs-official.svg)](https://pypi.org/project/pandas-stubs-official/)
[![Package Status](https://img.shields.io/pypi/status/pandas-stubs-official.svg)](https://pypi.org/project/pandas-stubs-official/)
[![License](https://img.shields.io/pypi/l/pandas.svg)](https://github.com/pandas-dev/pandas-stubs/blob/main/LICENSE)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<!-- [![Conda Latest Release](https://anaconda.org/conda-forge/pandas-stubs-official/badges/version.svg)](https://anaconda.org/anaconda/pandas-stubs-official/) -->
<!--[![Downloads](https://static.pepy.tech/personalized-badge/pandas?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/pandas-stubs-official) -->

## What is it?

This is the home for pandas typing stubs supported by the pandas core team.  The stubs are likely incomplete in terms of covering the published API of pandas.


## Documentation

The official documentation is hosted on: https://github.com/pandas-dev/pandas-stubs/tree/main/docs

- [How to set up the environment](docs/1%20-%20setup.md)
- [How to test the project](docs/2%20-%20tests.md)
- [How to follow the code style](docs/3%20-%20style.md)
- [Security stuffs](docs/4%20-%20security.md)
- [How to publish](docs/5%20-%20publish.md)

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/pandas-dev/pandas-stubs

Binary installers for the latest released version are still not avaible, but the goal is to be available on [Python
Package Index (PyPI)](https://pypi.org/project/pandas-stubs-official) and on [Conda](https://docs.conda.io/en/latest/).

```sh
# conda
conda install pandas-stubs-official
```

```sh
# or PyPI
pip install pandas-stubs-official
```

## Dependencies
- [Pandas - a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.](https://pandas.pydata.org/)



## Getting Help
For usage questions, the best place to go to is [StackOverflow](https://stackoverflow.com/questions/tagged/pandas).
Further, general questions and discussions can also take place on the [pydata mailing list](https://groups.google.com/forum/?fromgroups#!forum/pydata).

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

## Background

These stubs were forked from the project <https://github.com/microsoft/python-type-stubs> as of commit `6b800063bde687cd1846122431e2a729a9de625a`

As both projects move forward, this page will track what the differences are (if any).  There is an expectation that, in the near future, the Microsoft project will start pulling these stubs for inclusion in Visual Studio Code releases.


## Contributing to pandas [![Open Source Helpers](https://www.codetriage.com/pandas-dev/pandas/badges/users.svg)](https://www.codetriage.com/pandas-dev/pandas)

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the **[contributing guide](https://pandas.pydata.org/docs/dev/development/contributing.html)**.

If you are simply looking to start working with the pandas codebase, navigate to the [GitHub "issues" tab](https://github.com/pandas-dev/pandas/issues) and start looking through interesting issues. There are a number of issues listed under [Docs](https://github.com/pandas-dev/pandas/issues?labels=Docs&sort=updated&state=open) and [good first issue](https://github.com/pandas-dev/pandas/issues?labels=good+first+issue&sort=updated&state=open) where you could start out.

You can also triage issues which may include reproducing bug reports, or asking for vital information such as version numbers or reproduction instructions. If you would like to start triaging issues, one easy way to get started is to [subscribe to pandas on CodeTriage](https://www.codetriage.com/pandas-dev/pandas).

Or maybe through using pandas you have an idea of your own or are looking for something in the documentation and thinking ‘this can be improved’...you can do something about it!

Feel free to ask questions on the [mailing list](https://groups.google.com/forum/?fromgroups#!forum/pydata) or on [Gitter](https://gitter.im/pydata/pandas).

As contributors and maintainers to this project, you are expected to abide by pandas' code of conduct. More information can be found at: [Contributor Code of Conduct](https://github.com/pandas-dev/pandas/blob/main/.github/CODE_OF_CONDUCT.md)

## License
[BSD 3](LICENSE)

## Thanks

We are indebted to Microsoft and that project for the initial set of public type stubs.  We are also grateful for the original pandas-stubs project at <https://github.com/VirtusLab/pandas-stubs> that created the framework for testing the stubs.

Last update to README: 25/6/2022: 00:00 EDT
