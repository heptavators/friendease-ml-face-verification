import logging
import warnings
import tensorflow as tf

from loguru import logger

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

logger = logger

logger.add(
    "logs/history/{time:DD-MM-YYYY}.log",
    format="{time:DD-MM-YYYY at HH:mm:ss} | {level} | {message}",
    rotation="10 MB",
    level="DEBUG",
    colorize=True,
    enqueue=True,
)
