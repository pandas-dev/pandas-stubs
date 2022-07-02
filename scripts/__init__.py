import sys

from loguru import logger

# Config the format of log message
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
