import subprocess


def test_all():

    cmd = ["mypy", "pandas-stubs", "tests"]
    subprocess.run(cmd)

    cmd = ["pytest"]
    subprocess.run(cmd)

    cmd = ["pyright"]
    subprocess.run(cmd)
