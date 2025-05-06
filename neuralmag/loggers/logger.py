# SPDX-License-Identifier: MIT

import os
from collections.abc import Iterable

from ..common import logging
from .field_logger import FieldLogger
from .scalar_logger import ScalarLogger

__all__ = ["Logger"]


class Logger(object):
    """
    Combined Scalar- and Field-Logger class

    :param directory: The name of the log file
    :type directory: str
    :param scalars: List of state-attribute names of the scalars to be logged
    :type scalars: list
    :param fields: List of state-attribute names of the fields to be logged
    :type fields: list
    :param scalars_every: Write scalars every nth step
    :type scalars_every: int
    :param fields_every: Write fields every nth step
    :type fields_every: int

    :Example:
        .. code-block:: python

            # provide key strings with are available in state
            logger = Logger("data", ["m", "h_demag"], ["m"], fields_every=100)

            # Actually log fields
            state = State(mesh)
            logger.log(state)
    """

    def __init__(
        self, directory, scalars=[], fields=[], scalars_every=1, fields_every=1
    ):
        self.loggers = {}
        self._resume_time = None
        if len(scalars) > 0:
            self.loggers["scalars"] = ScalarLogger(
                os.path.join(directory, "log.dat"), scalars, every=scalars_every
            )
        if len(fields) > 0:
            self.loggers["fields"] = FieldLogger(
                os.path.join(directory, "fields.pvd"), fields, every=fields_every
            )

    def log(self, state):
        """
        Log simulation step

        :param state: The state to be logged
        :type state: :class:`State`
        """
        if state.t == self._resume_time:  # avoid logging directly after resume
            self._resume_time = None
            return
        for logger in self.loggers.values():
            logger.log(state)

    def resume(self, state):
        """
        Tries to resume from existing log files. If resume is possible,
        i.e. if at least one magnetization field has been logged before,
        the state object is updated with latest possible values of t
        and m and the different log files are aligned and resumed.
        If resume is not possible the state is not modified and the
        simulations starts from the beginning.

        :param state: The state to be resumed from
        :type state: :class:`State`
        """
        last_recorded_step = self.loggers["fields"].last_recorded_step()
        if last_recorded_step is None:
            logging.warning("Resume not possible. Start over.")
            return

        resumable_step = min(
            map(lambda logger: logger.resumable_step(), self.loggers.values())
        )
        assert resumable_step >= last_recorded_step + 1

        state.m, state.t = self.loggers["fields"].step_data(
            last_recorded_step, "m", state
        )
        logging.info_green(
            "Resuming from step %d (t = %g)." % (last_recorded_step, state.t)
        )

        for logger in self.loggers.values():
            logger.resume(last_recorded_step + 1)

        self._resume_time = state.t

    def is_resumable(self):
        """
        Returns True if logger can resume from log files.

        :return: True of resumable, False otherwise
        :rtype: bool
        """
        return self.loggers["fields"].last_recorded_step() is not None
