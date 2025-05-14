# SPDX-License-Identifier: MIT

import logging

__all__ = [
    "set_log_level",
    "debug",
    "warning",
    "error",
    "info",
    "info_green",
    "info_blue",
]

# create magnum.fe logger
logger = logging.getLogger("NeuralMag")

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s %(name)s:%(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(handler)

logger.setLevel(logging.INFO)

info = logger.info

RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"
CYAN = "\033[1;37;36m%s\033[0m"


def debug(message, *args, **kwargs):
    logger.debug(CYAN % message, *args, **kwargs)


def warning(message, *args, **kwargs):
    logger.warning(RED % message, *args, **kwargs)


def error(message, *args, **kwargs):
    logger.error(RED % message, *args, **kwargs)


def info_green(message, *args, **kwargs):
    info(GREEN % message, *args, **kwargs)


def info_blue(message, *args, **kwargs):
    info(BLUE % message, *args, **kwargs)


def set_log_level(level):
    """
    Set the log level of magnum.np specific logging messages.
    Defaults to :code:`INFO = 20`.

    *Arguments*
      level (:class:`int`)
        The log level
    """
    logger.setLevel(level)
