# TODO switch to Pathlib
import os
import numpy as np
import pyvista as pv
from . import Function, Mesh, State

__all__ = ['write_vti', 'read_vti']

def write_vti(fields, filename, state = None):
    if isinstance(fields, Function):
        fields = [fields]

    if state is None:
        state = fields[0].state

    n = state.mesh.n
    dx = state.mesh.dx
    origin = state.mesh.origin

    grid = pv.UniformGrid(dimensions=np.array(n) + 1, spacing=dx, origin=origin)

    for field in fields:
        if isinstance(field, str):
            name = field
            field = getattr(state, name)
        else:
            name = field.name

        if field.shape == ():
            data = field.tensor.detach().cpu().numpy().flatten("F")
        elif field.shape == (3,):
            data = field.tensor.detach().cpu().numpy().reshape(-1, 3, order="F")
        else:
            raise NotImplemented("Unsupported shape.")

        if field.ftype == "node":
            grid.point_data.set_array(data, name)
        elif field.ftype == "cell":
            grid.cell_data.set_array(data, name)
        else:
            raise NotImplemented("Unsupported ftype.")

    dirname = os.path.dirname(filename)
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname)

    grid.save(filename)

def read_vti(filename, name = None, state = None):
    fields = {}
    data = pv.read(filename)

    if state is None:
        mesh = Mesh(np.array(data.dimensions)-1, data.spacing, data.origin)
        state = State(mesh)
    else:
        assert state.mesh.n == np.array(data.dimesions)-1

    if name is None:
        name = data.array_names[0]

    if name in data.point_data.keys():
        ftype = 'node'
        n = tuple([x+1 for x in mesh.n])
    elif name in data.cell_data.keys():
        ftype = 'cell'
        n = mesh.n
    else:
        raise

    vals = data.get_array(name)
    if len(vals.shape) == 1:
        dim = n
        shape = ()
    else:
        dim = n + (vals.shape[-1],)
        shape = (3,)

    return Function(state, ftype = ftype, shape = shape, tensor = state.tensor(vals.reshape(dim, order="F")))
