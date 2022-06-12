import shutil
import subprocess
from pathlib import Path

from scripts._job import Step, run_job


def run_mypy_src():
    cmd = ["mypy", "pandas-stubs", "tests"]
    subprocess.run(cmd, check=True)


def run_pyright_src():
    cmd = ["pyright"]
    subprocess.run(cmd, check=True)


def run_pytest_src():
    cmd = ["pytest"]
    subprocess.run(cmd, check=True)


def test_src(profile):
    steps = []

    # Possible steps
    mypy_step = Step(name="Run Mypy Against Source Code", run=run_mypy_src)
    pyright_step = Step(name="Run Pyright Against Source Code", run=run_pyright_src)
    pytest_step = Step(name="Run Pytest Against Source Code", run=run_pytest_src)

    # Defining which test is going to run according to a profile
    if profile in (None, "", "default"):
        steps.append(mypy_step)
        steps.append(pyright_step)
    elif profile == "pytest":
        steps.append(Step(name="Run Pytest Against Source Code", run=run_pytest_src))
    elif profile == "full":
        steps.append(mypy_step)
        steps.append(pyright_step)
        steps.append(pytest_step)
    else:
        raise Exception("Profile not found!")

    run_job(steps)


def build_dist():
    cmd = ["poetry", "build", "-f", "wheel"]
    subprocess.run(cmd, check=True)


def install_dist():
    path = next(Path("dist/").glob("*.whl"))
    cmd = ["pip", "install", str(path)]
    subprocess.run(cmd, check=True)


def remove_src():
    shutil.rmtree(r"pandas-stubs")


def run_mypy_dist():
    cmd = ["mypy", "tests"]
    subprocess.run(cmd, check=True)


def run_pyright_dist():
    cmd = ["pyright", "tests"]
    subprocess.run(cmd, check=True)


def uninstall_dist():
    cmd = ["pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def restore_src():
    cmd = ["git", "checkout", "HEAD", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def install_poetry():
    cmd = ["poetry", "update", "-vvv"]
    subprocess.run(cmd, check=True)


def test_dist():
    steps = [
        Step(name="Build Dist", run=build_dist),
        Step(name="Install Dist", run=install_dist),
        Step(name="Remove Source Code", run=remove_src),
        Step(name="Run MyPy Against Dist", run=run_mypy_dist),
        Step(name="Run Pyright Against Dist", run=run_pyright_dist),
        Step(name="Uninstall Dist", run=uninstall_dist),
        Step(name="Restore Source Code", run=restore_src),
        Step(name="Install Poetry", run=install_poetry),
    ]

    run_job(steps)


def test_all():
    steps = [
        Step(name="Run Mypy Against Source Code", run=run_mypy_src),
        Step(name="Run Pyright Against Source Code", run=run_pyright_src),
        Step(name="Run Pytest Against Source Code", run=run_pytest_src),
        Step(name="Build Dist", run=build_dist),
        Step(name="Install Dist", run=install_dist),
        Step(name="Remove Source Code", run=remove_src),
        Step(name="Run MyPy Against Dist", run=run_mypy_dist),
        Step(name="Run Pyright Against Dist", run=run_pyright_dist),
        Step(name="Uninstall Dist", run=uninstall_dist),
        Step(name="Restore Source Code", run=restore_src)
    ]

    run_job(steps)
