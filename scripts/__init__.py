import sys

from loguru import logger
from pathlib import Path

pkg_path = [x for x in sys.path if x.endswith('site-packages')]

if not Path(fr'{pkg_path[0]}/my_paths.pth').exists():
    with open(fr'{pkg_path[0]}/my_paths.pth', 'w') as file:
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
