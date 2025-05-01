# MIT License
#
# Copyright (c) 2022-2025 NeuralMag team
#
# This file is part of NeuralMag – a simulation package for inverse micromagnetics.
# Repository: https://gitlab.com/neuralmag/neuralmag
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
