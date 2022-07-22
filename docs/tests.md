## Test

[Poe](https://github.com/nat-n/poethepoet) is used to run all tests.

Here are the most important options. Fore more details, please use `poe --help`.

- Run all tests (includes installing the stubs): `poe test_all`
- Run tests against the source code: `poe test_src` 
  - Run only mypy: `poe mypy_src`
  - Run only pyright: `poe pyright_src`
  - Run only pytest: `poe pytest`
  - Run only pre-commit: `poe style`
- Run tests against the installed stubs (this will install and uninstall the stubs): `poe test_dist`

These tests originally came from https://github.com/VirtusLab/pandas-stubs.
