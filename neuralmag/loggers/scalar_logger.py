# SPDX-License-Identifier: MIT

import os
from collections.abc import Iterable
from functools import reduce

from ..common import Function, config, logging

__all__ = ["ScalarLogger"]


class ScalarLogger(object):
    """
    Simple logger class to log scalar values into a tab separated file.

    :param filename: The name of the log file
    :type filename: str
    :param columns: List of attribute names to be logged
    :type columns: list
    :param every: Write row to log file every nth call
    :type every: int

    :Example:
        .. code-block:: python

            # provide key strings with are available in state
            logger = ScalarLogger("log.dat", ["t","m"])

            # provide func(state) or tuple (name, func(state))
            logger = ScalarLogger("log.dat", [("t[ns]", lambda state: state.t*1e9)])

            # Actually log a row
            state = State(mesh)
            logger.log(state)
    """

    def __init__(self, filename, columns, every=1):
        # create directory if not existent
        if not os.path.dirname(filename) == "" and not os.path.exists(
            os.path.dirname(filename)
        ):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self._filename = filename
        self._file = None
        self._every = every
        self._i = 0
        self._columns = columns

    def add_column(self, column):
        """
        Add column to log file.

        This method is automatically called for all
        columns on initialization of the logger.
        The column can be provided either as attribute name or as a Callable
        that takes the state as the only argument.
        If the column is a :class:`Function`, the functional is averaged over
        the whole mesh before logging.

        .. note::
          This method can only be called before the first line is logged

        :param column: The column to be logged
        :type column: str, Callable
        """
        if self._file is not None:
            raise RuntimeError(
                "You cannot add columns after first log row has been written."
            )
        self._columns.append(column)

    def log(self, state):
        """
        Log simulation step

        :param state: The state to be logged
        :type state: :class:`State`
        """
        self._i += 1
        if (self._i - 1) % self._every > 0:
            return

        values = []

        for column in self._columns:
            if isinstance(column, str):
                name = column
                raw_value = state.getattr(column)
            elif hasattr(column, "__call__"):
                name = column.__name__
                raw_value = column(state)
            elif isinstance(column, tuple) or isinstance(column, list):
                name = column[0]
                raw_value = column[1](state)
            else:
                raise RuntimeError("Column type not supported.")

            if isinstance(raw_value, Function):
                value = raw_value.avg().tolist()
            elif isinstance(raw_value, config.backend.Tensor):
                value = config.backend.to_numpy(raw_value).tolist()
            else:
                value = raw_value
            values.append((name, value))

        if self._file is None:
            self._file = open(self._filename, "w")
            self._write_header(values)

        self._write_row(values)

    def _write_header(self, columns):
        headings = []

        for column in columns:
            if isinstance(column[1], Iterable):
                if len(column[1]) == 3:
                    for i in ("x", "y", "z"):
                        headings.append(column[0] + "_" + i)
                else:
                    for i in range(len(column[1])):
                        headings.append(column[0] + "_" + str(i))
            else:
                headings.append(column[0])

        format_str = "#" + "    ".join(["%-22s"] * len(headings)) + "\n"
        self._file.write(format_str % tuple(headings))
        self._file.flush()

    def _write_row(self, columns):
        flat_values = reduce(
            lambda x, y: x + y,
            map(
                lambda x: tuple(x[1]) if isinstance(x[1], Iterable) else (x[1],),
                columns,
            ),
        )
        format_str = "    ".join(["%+.15e"] * len(flat_values)) + "\n"
        self._file.write(format_str % flat_values)
        self._file.flush()

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def resumable_step(self):
        """
        Returns the first step that can be written when resuming, e.g. if the
        logger logs every 10th step and the first (i = 0) step was already
        logged, the result is 10.

        :return: The step number
        :rtype: int
        """
        if self._file is not None:
            raise RuntimeError(
                "Cannot resume from log file that is already open for writing."
            )

        i = 0
        with open(self._filename, "r") as f:
            for i, l in enumerate(f):
                pass
        return i * self._every

    def resume(self, i):
        """
        Try to resume existing log file from log step i. The log file
        is truncated accordingly.

        :param i: The log step to resume from
        :type i: int
        """
        number = (self.resumable_step() - i) / self._every

        # from https://superuser.com/questions/127786/efficiently-remove-the-last-two-lines-of-an-extremely-large-text-file
        count = 0
        # with open(self._filename, 'r+b') as f:
        #    f.seek(0, os.SEEK_END)
        #    end = f.tell()
        #    while f.tell() > 0:
        #        f.seek(-1, os.SEEK_CUR)
        #        char = f.read(1)
        #        if char != '\n' and f.tell() == end:
        #            raise RuntimeError("Cannot resume: logfile does not end with a newline.")
        #        if char == '\n':
        #            count += 1
        #        if count == number + 1:
        #            f.truncate()
        #            break
        #        f.seek(-1, os.SEEK_CUR)

        with open(self._filename, "r+b", buffering=0) as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            while f.tell() > 0:
                f.seek(-1, os.SEEK_CUR)
                # print(f.tell())
                char = f.read(1)
                if char != b"\n" and f.tell() == end:
                    raise RuntimeError(
                        "Cannot resume: logfile does not end with a newline."
                    )
                    # print ("No change: file does not end with a newline")
                    # exit(1)
                if char == b"\n":
                    count += 1
                if count == number + 1:
                    f.truncate()
                    break
                    # print ("Removed " + str(number) + " lines from end of file")
                    # exit(0)
                f.seek(-1, os.SEEK_CUR)

        self._i = i
        if self._i > 0:
            self._file = open(self._filename, "a")
