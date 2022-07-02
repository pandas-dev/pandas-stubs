from pathlib import Path
import shutil
import subprocess
import sys


def create_mypy_pkg_file():
    pkg_path = [path for path in sys.path if path.endswith("site-packages")]

    if not Path(rf"{pkg_path[0]}/my_path.pth").exists():
        with open(rf"{pkg_path[0]}/my_path.pth", "w") as file:
            file.write(str(Path.cwd()))


def destroy_mypy_pkg_file():
    pkg_path = [path for path in sys.path if path.endswith("site-packages")]

    if Path(rf"{pkg_path[0]}/my_path.pth").exists():
        Path(rf"{pkg_path[0]}/my_path.pth").unlink()


def mypy_src():
    cmd = ["mypy", "pandas-stubs", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def pyright_src():
    cmd = ["pyright"]
    subprocess.run(cmd, check=True)


def pytest_src():
    cmd = ["pytest"]
    subprocess.run(cmd, check=True)


def build_dist():
    cmd = ["poetry", "build", "-f", "wheel"]
    subprocess.run(cmd, check=True)


def install_dist():
    path = next(Path("dist/").glob("*.whl"))
    cmd = ["pip", "install", str(path)]
    subprocess.run(cmd, check=True)


def rename_src():
    if Path(r"pandas-stubs").exists():
        Path(r"pandas-stubs").rename("_pandas-stubs")
    else:
        raise FileNotFoundError("'pandas-stubs' folder does not exists.")


def mypy_dist():
    cmd = ["mypy", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def pyright_dist():
    cmd = ["pyright", "tests"]
    subprocess.run(cmd, check=True)


def uninstall_dist():
    cmd = ["pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def restore_src():
    if Path(r"_pandas-stubs").exists():
        Path(r"_pandas-stubs").rename("pandas-stubs")
    else:
        raise FileNotFoundError("'_pandas-stubs' folder does not exists.")


def clean_mypy_cache():
    if Path(".mypy_cache").exists():
        shutil.rmtree(".mypy_cache")


def clean_pytest_cache():
    if Path(".mypy_cache").exists():
        shutil.rmtree(".pytest_cache")
