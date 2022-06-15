from scripts._job import Step, run_job
from scripts.test import procedures

def test_src(profile: str, clean_cache: bool = False):
    steps = []
    if clean_cache:
        steps.extend([
            Step(name="Clean mypy cache", run=procedures.clean_mypy_cache),
            Step(name="Clean pytest cache", run=procedures.clean_pytest_cache)
        ])

    # Possible steps
    mypy_step = Step(name="Run Mypy Against Source Code", run=procedures.run_mypy_src)
    pyright_step = Step(name="Run Pyright Against Source Code", run=procedures.run_pyright_src)
    pytest_step = Step(name="Run Pytest Against Source Code", run=procedures.run_pytest_src)

    # Defining which test is going to run according to a profile
    if profile in (None, "", "default"):
        steps.extend([mypy_step, pyright_step])
    elif profile == "pytest":
        steps.extend([pytest_step])
    elif profile == "full":
        steps.extend([mypy_step, pyright_step, pytest_step])
    else:
        raise Exception("Profile not found!")

    run_job(steps)


def test_dist(clean_cache: bool = False):
    steps = []
    if clean_cache:
        steps.extend([
            Step(name="Clean mypy cache", run=procedures.clean_mypy_cache),
            Step(name="Clean pytest cache", run=procedures.clean_pytest_cache)
        ])
    
    steps.extend([
        Step(name="Build Dist", run=procedures.build_dist),
        Step(name="Install Dist", run=procedures.install_dist, rollback=procedures.uninstall_dist),
        Step(name="Remove Source Code", run=procedures.remove_src, rollback=procedures.restore_src),
        Step(name="Run MyPy Against Dist", run=procedures.run_mypy_dist),
        Step(name="Run Pyright Against Dist", run=procedures.run_pyright_dist),
        Step(name="Uninstall Dist", run=procedures.uninstall_dist),
        Step(name="Restore Source Code", run=procedures.restore_src)
    ])

    run_job(steps)


def test_all(clean_cache: bool = False):
    steps = []
    if clean_cache:
        steps.extend([
            Step(name="Clean mypy cache", run=procedures.clean_mypy_cache),
            Step(name="Clean pytest cache", run=procedures.clean_pytest_cache)
        ])

    steps.extend([
        Step(name="Run Mypy Against Source Code", run=procedures.run_mypy_src),
        Step(name="Run Pyright Against Source Code", run=procedures.run_pyright_src),
        Step(name="Run Pytest Against Source Code", run=procedures.run_pytest_src),
        Step(name="Build Dist", run=procedures.build_dist),
        Step(name="Install Dist", run=procedures.install_dist, rollback=procedures.uninstall_dist),
        Step(name="Remove Source Code", run=procedures.remove_src, rollback=procedures.restore_src),
        Step(name="Run MyPy Against Dist", run=procedures.run_mypy_dist),
        Step(name="Run Pyright Against Dist", run=procedures.run_pyright_dist),
        Step(name="Uninstall Dist", run=procedures.uninstall_dist),
        Step(name="Restore Source Code", run=procedures.restore_src)
    ])

    run_job(steps)