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
