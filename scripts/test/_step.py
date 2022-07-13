from scripts._job import Step
from scripts.test import run

clean_mypy_cache = Step(name="Clean mypy cache", run=run.clean_mypy_cache)
clean_pytest_cache = Step(name="Clean pytest cache", run=run.clean_pytest_cache)
mypy_src = Step(
    name="Run mypy on 'tests' (using the local stubs) and on the local stubs",
    run=run.mypy_src,
)
pyright_src = Step(
    name="Run pyright on 'tests' (using the local stubs) and on the local stubs",
    run=run.pyright_src,
)
pytest_src = Step(name="Run pytest", run=run.pytest_src)
style_src = Step(name="Run pre-commit Against Source Code", run=run.style_src)
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
uninstall_dist = Step(name="Uninstall pandas-stubs", run=run.uninstall_dist)
restore_src = Step(name="Restore local stubs", run=run.restore_src)
