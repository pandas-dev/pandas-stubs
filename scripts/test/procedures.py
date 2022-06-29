import shutil
import subprocess
from pathlib import Path
import sys


def create_mypy_pkg_file():
    pkg_path = [path for path in sys.path if path.endswith('site-packages')]

    if not Path(fr'{pkg_path[0]}/my_path.pth').exists():
        with open(fr'{pkg_path[0]}/my_path.pth', 'w') as file:
            file.write(str(Path.cwd()))


def destroy_mypy_pkg_file():
    pkg_path = [path for path in sys.path if path.endswith('site-packages')]

    if Path(fr'{pkg_path[0]}/my_path.pth').exists():
        Path(fr'{pkg_path[0]}/my_path.pth').unlink()


def run_mypy_src():
    cmd = ["mypy", "pandas-stubs", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def run_pyright_src():
    cmd = ["pyright"]
    subprocess.run(cmd, check=True)


def run_pytest_src():
    cmd = ["pytest"]
    subprocess.run(cmd, check=True)


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
    cmd = ["mypy", "tests", "--no-incremental"]
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


def clean_mypy_cache():
    if Path('.mypy_cache').exists():
        shutil.rmtree('.mypy_cache')


def clean_pytest_cache():
    if Path('.mypy_cache').exists():
        shutil.rmtree('.pytest_cache')


def create_new_venv():
    cmd = ["poetry", "remove", "python"]
    subprocess.run(cmd, check=True)

    cmd = ["poetry", "update", "-vvv"]
    subprocess.run(cmd, check=True)

    cmd = ["poetry", "shell"]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    restore_last_changes()