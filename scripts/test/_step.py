from functools import partial

from scripts._job import Step
from scripts.test import run

ty_src = Step(
    name="Run ty on 'tests' (using the local stubs) and on the local stubs",
    run=partial(run.checker_src, "ty"),
)
ty_src_all = Step(
    name="Run ty on 'tests' (using the local stubs) and on the local stubs with all rules raising errors",
    run=partial(run.checker_src, "ty", all_rules=True),
)
pyrefly_src = Step(
    name="Run pyrefly on 'tests' (using the local stubs) and on the local stubs",
    run=partial(run.checker_src, "pyrefly"),
)
pyrefly_src_all = Step(
    name="Run pyrefly on 'tests' (using the local stubs) and on the local stubs with preset 'all'",
    run=partial(run.checker_src, "pyrefly", all_rules=True),
)
pyright_src = Step(
    name="Run pyright on 'tests' (using the local stubs) and on the local stubs",
    run=partial(run.checker_src, "pyright"),
)
mypy_src = Step(
    name="Run mypy on 'tests' (using the local stubs) and on the local stubs",
    run=partial(run.checker_src, "mypy"),
)
pytest = Step(name="Run pytest", run=run.pytest)
style = Step(name="Run pre-commit", run=run.style)
build_dist = Step(name="Build pandas-stubs", run=run.build_dist)
install_dist = Step(
    name="Install pandas-stubs", run=run.install_dist, rollback=run.uninstall_dist
)
rename_src = Step(
    name="Rename local stubs",
    run=run.rename_src,
    rollback=run.restore_src,
)
mypy_dist = Step(
    name="Run mypy on 'tests' using the installed stubs",
    run=partial(run.checker_dist, "mypy"),
)
pyright_dist = Step(
    name="Run pyright on 'tests' using the installed stubs",
    run=partial(run.checker_dist, "pyright"),
)
pyrefly_dist = Step(
    name="Run pyrefly on 'tests' using the installed stubs",
    run=partial(run.checker_dist, "pyrefly"),
)
ty_dist = Step(
    name="Run ty on 'tests' using the installed stubs",
    run=partial(run.checker_dist, "ty"),
)
stubtest = Step(
    name="Run stubtest to compare the installed stubs against pandas", run=run.stubtest
)
nightly = Step(
    name="Install pandas nightly",
    run=partial(
        run.install_latest,
        "pandas",
        extra_index_url="https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
    ),
    rollback=partial(run.install_floor, "pandas"),
)
mypy_nightly = Step(
    name="Install mypy nightly", run=run.nightly_mypy, rollback=run.released_mypy
)
pyrefly_pre_release = Step(
    name="Install the newest pyrefly (pre-releases included)",
    run=partial(run.install_latest, "pyrefly"),
    rollback=partial(run.install_floor, "pyrefly"),
)
