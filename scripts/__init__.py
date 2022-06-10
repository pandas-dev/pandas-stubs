import sys

from loguru import logger

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
