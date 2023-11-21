from loguru import logger

logger = logger

logger.add(
    "logs/history/{time:DD-MM-YYYY}.log",
    format="{time:DD-MM-YYYY at HH:mm:ss} | {level} | {message}",
    rotation="10 MB",
    level="DEBUG",
    colorize=True,
    enqueue=True,
)
