"""
NeuralMag - A nodal finite-difference code for inverse micromagnetics

Copyright (c) 2024 NeuralMag team

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser Python General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser Python General Public License for more details.

You should have received a copy of the Lesser Python General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
