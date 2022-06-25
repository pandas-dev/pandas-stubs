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


## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/pandas-dev/pandas-stubs

Binary installers for the latest released version are still not avaible, but the future goal is to be on [Python
Package Index (PyPI)](https://pypi.org/project/pandas-stubs-official) and on [Conda](https://docs.conda.io/en/latest/).

```sh
# conda
conda install pandas-stubs-official
```

```sh
# or PyPI
pip install pandas-stubs-official
```

After installed, you already could enjoy the new type annotations.

## Development

You can easily help the project development. If you saw something that does not make sense in our documentation, or you found a new issue. Here how you can contribute:

### Set up your environment

- Make sure you have `python ">=3.8", "<3.11"` installed.
- Install poetry if you still don't have:  `pip install poetry`.
- Install the project dependencies with: `poetry update -vvv`.
- Enter the virtual environment: `poetry shell`.

More details on [Setup Docs](https://github.com/pandas-dev/pandas-stubs/blob/main/docs/1%20-%20setup.md)

### Develop a feature

- Run the  local style test to see if the project is following the black and isort style: `poe check_style`. 
- Run all stubs tests to check if your current project is ok: `poe test_all`.
- Create a new feature in the stubs files.
- Write a new test to validate what you've implemented. New PR must include an appropriate test. See `pandas-stubs\tests` for examples.

More details on [Tests Docs](https://github.com/pandas-dev/pandas-stubs/blob/main/docs/2%20-%20tests.md) or [Code Style Docs](https://github.com/pandas-dev/pandas-stubs/blob/main/docs/3%20-%20style.md)

### Submit your Pull Request

- Pull your changes to your forked project.
- Wait the CI/CD pipeline runs
- If everything is ok you can finally make your PR, make sure to describe what the new feature is about.

More details on [CI/CD Docs](https://github.com/pandas-dev/pandas-stubs/blob/main/docs/2%20-%20tests.md)


## Background

These stubs were forked from the project <https://github.com/microsoft/python-type-stubs> as of commit `6b800063bde687cd1846122431e2a729a9de625a`

As both projects move forward, this page will track what the differences are (if any).  There is an expectation that, in the near future, the Microsoft project will start pulling these stubs for inclusion in Visual Studio Code releases.


## Thanks

We are indebted to Microsoft and that project for the initial set of public type stubs.  We are also grateful for the original pandas-stubs project at <https://github.com/VirtusLab/pandas-stubs> that created the framework for testing the stubs.

Last update to README: 25/6/2022: 00:00 EDT
