# SPDX-License-Identifier: MIT

import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
from xml.etree import cElementTree

__all__ = ["FieldLogger"]


class FieldLogger(object):
    """
    Logger class for fields using PVD/VTI files from the visualization
    toolkit (VTK). The logger creates a single PVD file referencing the
    VTI snapshots for different simulation times saving spatially
    resolved field data.

    :param filename: The name of the log file
    :type filename: str
    :param fields: The fields to be written to the log file as a list of
                   attribute names of the state class
    :type fields: list
    :param every: Write field to log file every nth call
    :type every: int, optional

    :Example:
        .. code-block:: python

            # provide key strings which are available in state
            logger = FieldLogger("data/m.pvd", ["m", "h_demag"])

            # Actually log fields
            state = State(mesh)
            logger.log(state)
    """

    def __init__(self, filename, fields, every=1):
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
        """
        Log simulation step

        :param state: The state to be logged
        :type state: :class:`State`
        """
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

    def reset(self):
        """
        Reset the internal step counter
        """
        self._i = 0

    def resumable_step(self):
        """
        Returns the first step that can be written when resuming, e.g. if the
        logger logs every 10th step and the first (i = 0) step was already
        logged, the result is 10.

        :return: The step number
        :rtype: int
        """
        try:
            xml = cElementTree.parse(self._filename + ".pvd").getroot()
            return len(list(xml.find("Collection"))) * self._every
        except IOError:
            return 0

    def last_recorded_step(self):
        """
        Returns the number of the last step logged and ``None`` if no
        step was yet logged.

        :return: Number of the last step recorded
        :rtype: int/None
        """
        result = (self.resumable_step() // self._every - 1) * self._every
        if result < 0:
            return None
        else:
            return result

    def step_data(self, i, field, state):
        """
        Returns field and time to a given step number.

        :param i: The step number
        :type i: int
        :param field: The name of the field to be read
        :type field: str
        :param state: The state used for the read
        :type state: :class:`State`
        :return: The field as a function
        :rtype: :class:`Function`
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

        :param i: The log step to resume from
        :type i: int
        """
        self._i = i
        self._i_start = self.last_recorded_step() + 1
        self._xmlroot = cElementTree.parse(self._filename + ".pvd").getroot()
