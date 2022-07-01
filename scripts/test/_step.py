from scripts._job import Step
from scripts.test import run

clean_mypy_cache = Step(name="Clean mypy cache", run=run.clean_mypy_cache)
clean_pytest_cache = Step(name="Clean pytest cache", run=run.clean_pytest_cache)
mypy_src = Step(name="Run Mypy Against Source Code", run=run.mypy_src)
pyright_src = Step(name="Run Pyright Against Source Code", run=run.pyright_src)
pytest_src = Step(name="Run Pytest Against Source Code", run=run.pytest_src)
build_dist = Step(name="Build Dist", run=run.build_dist)
install_dist = Step(
    name="Install Dist", run=run.install_dist, rollback=run.uninstall_dist
)
rename_src = Step(
    name="Rename Source Code Folder",
    run=run.rename_src,
    rollback=run.restore_src,
)
mypy_dist = Step(name="Run MyPy Against Dist", run=run.mypy_dist)
pyright = Step(name="Run Pyright Against Dist", run=run.pyright_dist)
uninstall_dist = Step(name="Uninstall Dist", run=run.uninstall_dist)
restore_src = Step(name="Restore Source Code", run=run.restore_src)
