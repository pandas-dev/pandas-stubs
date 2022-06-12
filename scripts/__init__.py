import sys

from loguru import logger
from pathlib import Path

pkg_path = [x for x in sys.path if x.endswith('site-packages')]
with open(fr'{pkg_path[0]}/my_paths.pth', 'w') as file:
    file.write(str(Path.cwd()))


config = {
    "handlers": [
        {
            "sink": sys.stderr,
            "format": (
                "\n<level>===========================================</level>\n"
                "<level>{message}</level>\n"
                "<level>===========================================</level>\n"
            ),
        }
    ]
}

logger.configure(**config)
