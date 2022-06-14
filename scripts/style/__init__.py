from scripts.style import procedures
from scripts._job import Step, run_job

def check_style():
    steps = [Step(name="Check Black Style Code", run=procedures.run_black_check),
             Step(name="Check Isort Style Code", run=procedures.run_isort_check)]

    run_job(steps)