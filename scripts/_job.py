import time
from loguru import logger
from dataclasses import dataclass
from typing import Callable, List
from subprocess import CalledProcessError


@dataclass
class Step:
    name: str
    run: Callable[[], None]


def run_job(steps: List[Step]) -> None:
    """
        Responsible to run procedures with logs
    """

    for step in steps:
        start = time.perf_counter()
        logger.info(f"Beginning to run: '{step.name}'")

        try:
            step.run()
        except CalledProcessError:
            logger.error(f"'{step.name}' failed!")
            break

        end = time.perf_counter()
        logger.success(f"End '{step.name}', runtime: {end - start:.3f} seconds.")