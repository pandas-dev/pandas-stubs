## Test

[Poe](https://github.com/nat-n/poethepoet) is used to run all tests.

Here are the most important options. Fore more details, please use `poe --help`.

- Run all tests (against both source and installed stubs): `poe test_all`
- Run tests against the source code: `poe test` 
  - Run only mypy: `poe mypy`
  - Run only pyright: `poe pyright`
  - Run only pytest: `poe pytest`
  - Run only pre-commit: `poe style`
- Run tests against the installed stubs (this will install and uninstall the stubs): `poe test_dist`

These tests originally came from https://github.com/VirtusLab/pandas-stubs.

The following tests are **optional**. Some of them are run by the CI but it is okay if they fail.

- Run pytest against pandas nightly: `poe pytest --nightly`
- Use mypy nightly to validate the annotations: `poe mypy --mypy_nightly`
- Run stubtest to compare the installed pandas-stubs against pandas (this will fail): `poe stubtest`. If you have created an allowlist to ignore certain errors: `poe stubtest path_to_the_allow_list`
