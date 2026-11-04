## Test

[Poe](https://github.com/nat-n/poethepoet) is used to run all tests.

Here are the most important options. Fore more details, please use `poe --help`.

- Run all tests (against both source and installed stubs): `poe test_all`
- Run tests against the source code: `poe test`
  - Run only mypy: `poe mypy`
  - Run only pyright: `poe pyright`
  - Run only pyrefly: `poe pyrefly`
  - Run only ty: `poe ty`
  - Run runtime tests (excluding architecture tests): `poe pytest`
  - Run operand hierarchy architecture tests: `poe pytest_architecture`
  - Run only pre-commit: `poe style`
- Run tests against the installed stubs (this will install and uninstall the stubs): `poe test_dist`
- Verify type completeness: `poe type_completeness`.
- Run stubtest against pandas with the committed allowlists (a required CI job on Python 3.15): `poe stubtest`. See the headers of `scripts/test/stubtest-allowlist-permanent.txt` and `scripts/test/stubtest-allowlist-burndown.txt`. Use `poe stubtest --no_allowlist` to report every mismatch.

Some of these tests originally came from https://github.com/VirtusLab/pandas-stubs.

The following tests are **optional**. Some of them are run by the CI but it is okay if they fail.
The pandas nightly, mypy nightly and pyrefly pre-release tests run once a day on `main` (`.github/workflows/nightly.yml`).

- Run pytest against pandas nightly: `poe pytest --nightly`
- Use mypy nightly to validate the annotations: `poe mypy --mypy_nightly`
- Use the newest pyrefly on PyPI, pre-releases included, to validate the annotations: `poe pyrefly --pyrefly_pre_release`
- Use pyrefly with preset 'all': `poe pyrefly_all`
- Use ty with [all rules raising errors](https://docs.astral.sh/ty/rules/#rule-levels): `poe ty_all`

`--pyrefly_pre_release` installs with `pip install --pre --upgrade`. `--pre` selects the highest PEP 440 version rather than requiring a pre-release, so during a development cycle it is the `.devN` of the next pyrefly series and, once that series ships, it is the newest stable release.

Among the tests above, the following can be run directly during a PR by commenting in the discussion.

- Run pytest against pandas nightly by commenting `/pandas_nightly`
- Use mypy nightly to validate the annotations by commenting `/mypy_nightly`
- Use the newest pyrefly on PyPI, pre-releases included, to validate the annotations by commenting `/pyrefly_pre_release`
- Use pyrefly with preset 'all': `/pyrefly_all`
- Use ty with all rules raising errors: `/ty_all`

## Comparison test ownership

Keep cross-container comparison tests with the higher-tier container in the operand
hierarchy, following the existing arithmetic test organization. For example,
Series/Index comparisons belong in `tests/series/<dtype>/test_relops.py`. Cover both
operand directions together there, and cover the Series `lt`, `le`, `gt`, and `ge`
methods. Index-only comparisons belong in `tests/indexes/<dtype>/test_relops.py`.
When both operands have the same type, swapping them does not add typing coverage.
