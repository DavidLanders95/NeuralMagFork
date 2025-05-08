# SPDX-License-Identifier: MIT

from functools import reduce

from neuralmag.common import logging

__all__ = ["Mesh"]


class Mesh(object):
    r"""
    Regular cuboid mesh for nodal finite-differences. If number of cells in
    all principal directions are provided, a 3D mesh is created with
    :math:`(n_1 + 1) \times (n_2 + 1) \times (n_3 + 1)` cells.
    If only :math:`n_1` and :math:`n_2` is provided, a 2D mesh is created
    with a single layer of nodes, but finite thickness.
    If only :math:`n_1` is provided, a 1D mesh is created with a single line
    of nodes, but finite thickness and width.

    :param n: Number of cells in principal directions
    :type n: tuple
    :param dx: Cell-size in principal directions in m
    :type dx: tuple
    :param origin: Coordinate of the bottom-left corner of the mesh
    :type origin: tuple

    :Example:
        .. code-block::

            # 3D with 1 cell thickness, leading to 2 nodes in z-direction
            mesh_3d = Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))

            # 2D with 1 cell thickness, leading to 1 nodes in z-direction
            mesh_2d = Mesh((100, 25), (5e-9, 5e-9, 3e-9))
    """

    def __init__(self, n, dx, origin=(0, 0, 0)):
        self.n = tuple(n)
        self.dim = len(n)
        self.dx = tuple(dx)
        self.origin = tuple(origin)
        logging.info_green(
            f"[Mesh] {self.dim}D, {' x '.join([str(x) for x in self.n])} (size = {' x '.join(['{:,g}'.format(x) for x in self.dx])})"
        )

    @property
    def cell_volume(self):
        """
        The cell volume
        """
        return self.dx[0] * self.dx[1] * self.dx[2]

    @property
    def volume(self):
        """
        The total volume of the mesh
        """
        return self.num_cells * self.cell_volume

    @property
    def num_cells(self):
        """
        The total number of simulation cells
        """
        return reduce(lambda x, y: x * y, self.n)

    @property
    def num_nodes(self):
        """
        The total number of simulation nodes
        """
        # TODO better use a loop here?
        if self.dim == 3:
            return (self.n[0] + 1) * (self.n[1] + 1) * (self.n[2] + 1)
        elif self.dim == 2:
            return (self.n[0] + 1) * (self.n[1] + 1)
        elif self.dim == 1:
            return self.n[0] + 1
        else:
            raise RuntimeError(f"Mesh dimension '{self.dim}' not supported")

    def __str__(self):
        return f"{'x'.join(str(x) for x in self.n)}_{self.dx[0]:g}x{self.dx[1]:g}x{self.dx[2]:g}"
