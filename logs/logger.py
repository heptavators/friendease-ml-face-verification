import logging
import sys
import warnings
import tensorflow as tf

from loguru import logger

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

logger = logger

logger.add(
    sys.stdout,
    format="{time:DD-MM-YYYY at HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    colorize=True,
    enqueue=True,
)
