## Test

[Poe](https://github.com/nat-n/poethepoet) is used to run all tests.

Here are the most important options. Fore more details, please use `poe --help`.

- Run all tests (against both source and installed stubs): `poe test_all`
- Run tests against the source code: `poe test`
  - Run only mypy: `poe mypy`
  - Run only pyright: `poe pyright`
  - Run only pyrefly: `poe pyrefly`
  - Run only ty: `poe ty`
  - Run only pytest: `poe pytest`
  - Run only pre-commit: `poe style`
- Run tests against the installed stubs (this will install and uninstall the stubs): `poe test_dist`
- Verify type completeness: `poe type_completeness`.

Some of these tests originally came from https://github.com/VirtusLab/pandas-stubs.

The following tests are **optional**. Some of them are run by the CI but it is okay if they fail.

- Run pytest against pandas nightly: `poe pytest --nightly`
- Use mypy nightly to validate the annotations: `poe mypy --mypy_nightly`
- Use pyrefly with preset 'all': `poe pyrefly_all`
- Use ty with [all rules raising errors](https://docs.astral.sh/ty/rules/#rule-levels): `poe ty_all`
- Run stubtest to compare the installed pandas-stubs against pandas: `poe stubtest` passes against the committed burn-down allowlist `scripts/test/stubtest-allowlist.txt` (targeting the Python 3.14 CI baseline). Use `poe stubtest --no_allowlist` to report every mismatch, or `poe stubtest path_to_the_allow_list` to use a different allowlist.

Among the tests above, the following can be run directly during a PR by commenting in the discussion.

- Run pytest against pandas nightly by commenting `/pandas_nightly`
- Use mypy nightly to validate the annotations by commenting `/mypy_nightly`
- Use pyrefly with preset 'all': `/pyrefly_all`
- Use ty with all rules raising errors: `/ty_all`
- Run stubtest against the committed allowlist: `/stubtest`
