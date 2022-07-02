from scripts._job import (
    Step,
    run_job,
)
from scripts.style import procedures


def check_style():
    steps = [
        Step(name="Check Pyupgrade Style Code", run=procedures.run_pyupgrade_check),
        Step(name="Check Black Style Code", run=procedures.run_black_check),
        Step(name="Check Isort Style Code", run=procedures.run_isort_check),
    ]

    run_job(steps)


def format_style():
    steps = [
        Step(name="Format Pyupgrade Style Code", run=procedures.run_format_pyupgrade),
        Step(name="Format Black Style Code", run=procedures.run_format_black),
        Step(name="Format Isort Style Code", run=procedures.run_format_isort),
    ]

    run_job(steps)
