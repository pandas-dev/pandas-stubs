from scripts._job import Step
from scripts.test import run

mypy_src = Step(
    name="Run mypy on 'tests' (using the local stubs) and on the local stubs",
    run=run.mypy_src,
)
ty_src = Step(
    name="Run ty on 'pandas-stubs' (using the local stubs) and on the local stubs",
    run=run.ty,
)
pyrefly_src = Step(
    name="Run pyrefly on the local stubs",
    run=run.pyrefly,
)
pyright_src = Step(
    name="Run pyright on 'tests' (using the local stubs) and on the local stubs",
    run=run.pyright_src,
)
pyright_src_strict = Step(
    name="Run pyright on 'tests' (using the local stubs) and on the local stubs in full strict mode",
    run=run.pyright_src_strict,
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
    name="Run mypy on 'tests' using the installed stubs", run=run.mypy_dist
)
pyright_dist = Step(
    name="Run pyright on 'tests' using the installed stubs", run=run.pyright_dist
)
stubtest = Step(
    name="Run stubtest to compare the installed stubs against pandas", run=run.stubtest
)
nightly = Step(
    name="Install pandas nightly", run=run.nightly_pandas, rollback=run.released_pandas
)
mypy_nightly = Step(
    name="Install mypy nightly", run=run.nightly_mypy, rollback=run.released_mypy
)
