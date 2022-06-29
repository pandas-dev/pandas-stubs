import sys
from pathlib import Path

from loguru import logger

pkg_path = [x for x in sys.path if x.endswith("site-packages")]

if not Path(rf"{pkg_path[0]}/my_paths.pth").exists():
    with open(rf"{pkg_path[0]}/my_paths.pth", "w") as file:
        file.write(str(Path.cwd()))


config = {
    "handlers": [
        {
            "sink": sys.stderr,
            "format": (
                "<level>\n"
                "===========================================\n"
                "{message}\n"
                "===========================================\n"
                "</level>"
            ),
        }
    ]
}

logger.configure(**config)
