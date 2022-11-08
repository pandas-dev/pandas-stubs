## Test

[Poe](https://github.com/nat-n/poethepoet) is used to run all tests.

Here are the most important options. Fore more details, please use `poe --help`.

- Run all tests (against both source and installed stubs): `poe test_all`
- Run tests against the source code: `poe test` 
  - Run only mypy: `poe mypy`
  - Run only pyright: `poe pyright`
  - Run only pytest: `poe pytest`
  - Run only pre-commit: `poe style`
  - Run mypy and generate a report of the code coverage from the tests: `poe mypy_coverage`
- Run tests against the installed stubs (this will install and uninstall the stubs): `poe test_dist`
- Optional: run pytest against pandas nightly (this might fail): `poe pytest --nightly`
- Optional: Run stubtest to compare the installed pandas-stubs against pandas (this will fail): `poe stubtest`. If you have created an allowlist to ignore certain errors: `poe stubtest path_to_the_allow_list`
- Optional: Run mypy with coverage: `poe mypy --coverage`. The reported coverage documents functions thay appear to be fully, 
  partially, or untyped.  The coverage report is a good place to start when looking for imprecise types to improve, or when
  adding classes and functions to pandas-stubs.

## Credits
These tests originally came from https://github.com/VirtusLab/pandas-stubs.
