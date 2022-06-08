"""
Setup for Pandas type annotations.
For the sake of convenience the package installation
will coexist with Pandas installation.
If this is a problem - download the source
and add it to PYTHONPATH manually.
"""

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from typing import Generator
import os, shutil

version = "1.4.2.220604"


# find_packages might not work with stub files
src_path = "typings"


class BuildPyCommand(build_py):
    """Custom build command."""

    def run(self):
        shutil.rmtree("src.tmp", ignore_errors=True)
        os.mkdir("src.tmp")
        shutil.copytree("typings/pandas", "src.tmp/pandas-stubs")
        build_py.run(self)


def list_packages(source_path: str = src_path) -> Generator:
    for root, _, _ in os.walk(os.path.join(source_path, "pandas")):
        yield ".".join(os.path.relpath(root, source_path).split(os.path.sep)).replace(
            "pandas", "pandas-stubs"
        )


setup(
    cmdclass={"build_py": BuildPyCommand},
    package_dir={"": "src.tmp"},
    version=version,
    packages=list(list_packages()),
    package_data={"pandas-stubs": ["*.pyi", "**/*.pyi"]},
    install_requires=['typing_extensions>=4.2;python_version>="3.8"'],
)
