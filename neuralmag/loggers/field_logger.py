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

import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
from xml.etree import cElementTree

__all__ = ["FieldLogger"]


class FieldLogger(object):
    def __init__(self, filename, fields, every=1):
        """
        Logger class for fields

        *Arguments*
            filename (:class:`str`)
                The name of the log file
            fields ([:class:`str` | :class:`function`])
                The columns to be written to the log file
            every (:class:`int`)
                Write row to log file every nth call

        *Example*
            .. code-block:: python

                # provide key strings with are available in state
                logger = FieldLogger('data/m.pvd', ['m', demag.h])

                # Actually log fields
                state = State(mesh)
                logger << state
        """
        # create directory if not existent
        if not os.path.dirname(filename) == "" and not os.path.exists(
            os.path.dirname(filename)
        ):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        filename, ext = os.path.splitext(filename)
        if ext != ".pvd":
            raise NameError("Only .pvd extention allowed")
        self._filename = filename
        self._every = every
        if isinstance(fields, str):
            fields = [fields]
        self._fields = fields
        self._i = 0
        self._i_start = 0
        self._xmlroot = cElementTree.Element(
            "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
        )
        cElementTree.SubElement(self._xmlroot, "Collection")

    def log(self, state):
        self._i += 1
        if (self._i - 1) % self._every > 0:
            return
        if self._i <= self._i_start:
            return

        filename = "%s_%04d.vti" % (self._filename, self._i // self._every)
        state.write_vti(self._fields, filename)
        cElementTree.SubElement(
            self._xmlroot[0],
            "DataSet",
            timestep=str(state.t.tolist()),
            file=os.path.basename(filename),
        )
        with open(self._filename + ".pvd", "w") as fd:
            fd.write(
                minidom.parseString(
                    " ".join(
                        cElementTree.tostring(self._xmlroot)
                        .decode()
                        .replace("\n", "")
                        .split()
                    ).replace("> <", "><")
                ).toprettyxml(indent="  ")
            )

    def __lshift__(self, state):
        self.log(state)

    def reset(self):
        self._i = 0

    def resumable_step(self):
        try:
            xml = cElementTree.parse(self._filename + ".pvd").getroot()
            return len(list(xml.find("Collection"))) * self._every
        except IOError:
            return 0

    def last_recorded_step(self):
        """
        Returns the number of the last step logged and None if no
        step was yet logged.

        *Returns*
            :class:`int`
                Number of the last step recorded
        """
        result = (self.resumable_step() // self._every - 1) * self._every
        if result < 0:
            return None
        else:
            return result

    def step_data(self, i, field, state):
        """
        Returns field and time to a given step number.

        *Arguments*
            i (:class:`int`)
                The step number
            field (:class:`str`)
                The field to be read
            state (:class:`State`)
                The state to be used in the field function

        *Returns*
            (:class:`dolfin.Function`, :class:`float`)
                The field of step i and the corresponding time
        """
        if i % self._every > 0:
            raise Exception()

        xml = cElementTree.parse(self._filename + ".pvd").getroot()
        item = list(xml.find("Collection"))[i // self._every]
        field = state.read_vti(
            os.path.join(os.path.dirname(self._filename), item.attrib["file"]), field
        )
        return field, float(item.attrib["timestep"])

    def resume(self, i):
        """
        Try to resume existing log file from log step i. The log file
        is truncated accordingly.

        *Arguments*
            i (:class:`int`)
                The log step to resume from
        """
        self._i = i
        self._i_start = self.last_recorded_step() + 1
        self._xmlroot = cElementTree.parse(self._filename + ".pvd").getroot()
