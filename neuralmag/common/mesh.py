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

from . import logging

__all__ = ["Mesh"]


class Mesh(object):
    def __init__(self, n, dx, origin=(0, 0, 0)):
        self.n = tuple(n)
        self.dim = len(n)
        self.dx = tuple(dx)
        self.origin = tuple(origin)

    @property
    def cell_volume(self):
        return self.dx[0] * self.dx[1] * self.dx[2]

    @property
    def volume(self):
        return self.n[0] * self.n[1] * self.n[2] * self.cell_volume

    @property
    def num_cells(self):
        return self.n[0] * self.n[1] * self.n[2]

    @property
    def num_nodes(self):
        return (self.n[0] + 1) * (self.n[1] + 1) * (self.n[2] + 1)

    def __str__(self):
        return "%dx%dx%d_%gx%gx%g" % (self.n + self.dx)
