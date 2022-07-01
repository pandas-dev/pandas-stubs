import subprocess


def check_black():
    cmd = ["black", "--check", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)


def check_isort():
    cmd = ["isort", "--check-only", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)


def format_black():
    cmd = ["black", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)


def format_isort():
    cmd = ["isort", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)
