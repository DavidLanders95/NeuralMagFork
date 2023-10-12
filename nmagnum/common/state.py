import os
import numpy as np
import torch
import inspect
import types
import pyvista as pv
from . import logging
from . import Function

__all__ = ['State']

class Material:
    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        return getattr(self._state, "material__" + name)

    def __setattr__(self, name, value):
        # don't mess with protected attributes
        if name[0] == '_':
            super().__setattr__(name, value) 
            return
        return setattr(self._state, "material__" + name, value)

class State(object):
    def __init__(self, mesh, device=None):
        self._attr_values = {}
        self._attr_types = {}
        self._attr_funcs = {}
        self._attr_args = {}

        if device == None:
            CUDA_DEVICE = os.environ.get("CUDA_DEVICE", "0")
            self._device = torch.device(
                f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = device

        self._material = Material(self)
        self._mesh = mesh
        self.dx = self.tensor(mesh.dx)
        self.t = 0.

        logging.info_green(f"[State] Running on device: {self._device}")
        if mesh.dim == 2:
            logging.info_green("[Mesh] 2D, %dx%d (size = %g x %g x %g)" % (mesh.n + mesh.dx))
        elif mesh.dim == 3:
            logging.info_green("[Mesh] 3D, %dx%dx%d (size = %g x %g x %g)" % (mesh.n + mesh.dx))
        else:
            raise

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return torch.float64

    @property
    def mesh(self):
        return self._mesh

    @property
    def material(self):
        return self._material

    def getattr(self, name):
        container = self
        while '.' in name:
            parent, child = name.split('.', 1)
            container = getattr(container, parent)
            name = child
        return getattr(container, name)

    def tensor(self, value):
        if isinstance(value, torch.Tensor):
            # TODO check dtype and device?
            return value
        #if isinstance(value, np.ndarray):
        #    return torch.from_numpy(value).to(device = self.device, dtype = self.dtype)
        return torch.tensor(value, device = self.device, dtype = self.dtype, requires_grad = False)

    def __getattr__(self, name):
        if callable(self._attr_values[name]):
            if not name in self._attr_funcs:
                attr = self._attr_values[name]
                self._attr_funcs[name] = self.get_func(attr)
            func, args = self._attr_funcs[name]

            if name in self._attr_types:
                ftype, shape = self._attr_types[name]
                return Function(self, ftype = ftype, shape = shape, tensor = func(*args))
            else:
                return func(*args)
        else:
            return self._attr_values[name]

    def __setattr__(self, name, value): 
        # don't mess with protected attributes
        if name[0] == '_':
            super().__setattr__(name, value) 
            return  

        if isinstance(value, (int, float)):
            if name in self._attr_values:
                self._attr_values[name].fill_(value)
                return
            value = self.tensor(value)

        if isinstance(value, tuple) and len(value) == 3:
            self._attr_types[name] = value[1:]
            value = value[0]
        else:
            self._attr_types.pop(name, None)

        self._attr_values[name] = value
        self._attr_funcs.clear()
        self._attr_args.clear()

    def _collect_func_deps(self, attr):
        func_names = []
        args = {}
        for arg in list(inspect.signature(attr).parameters.keys()):
            attr = self._attr_values[arg]

            if callable(attr):
                func_names.append(arg)
                subfunc_names, subargs = self._collect_func_deps(attr)
                func_names = [f for f in func_names if f not in subfunc_names] + subfunc_names
                args.update(subargs)
            else:
                args[arg] = attr

        return func_names, args

    def get_func(self, f, add_args = []):
        func_names, args = self._collect_func_deps(f)
        args = list(set(args) - set(add_args))
        name = f.__name__
        name = "lmda" if f.__name__ == '<lambda>' else name

        # setup function with all dependencies
        if func_names:
            code = f"def {name}({', '.join(add_args + sorted(args))}):\n"
            func_pointers= {}
            for func_name in reversed(func_names):
                func = self._attr_values[func_name]
                func_pointers[f"__{func_name}"] = func
                code += f"    {func_name} = __{func_name}({', '.join(list(inspect.signature(func).parameters.keys()))})\n"
            func_pointers[f"__{name}"] = f
            code += f"    return __{name}({', '.join(list(inspect.signature(f).parameters.keys()))})\n"
            compiled_code = compile(code, "<string>", "exec")
            func = types.FunctionType(compiled_code.co_consts[0], func_pointers, name)
        else:
            func = f

        # collect args
        args = []
        for arg in list(inspect.signature(func).parameters.keys()):
            attr = self.getattr(arg.replace('__', '.'))
            if hasattr(attr, 'tensor'):
                args.append(attr.tensor)
            else:
                args.append(attr)

        return func, args

    @staticmethod
    def wrap_func(f, mapping):
        name = "lmda" if f.__name__ == '<lambda>' else f.__name__
        old_args = list(inspect.signature(f).parameters.keys())
        new_args = [mapping.get(a, a) for a in old_args]

        code = f"def {name}({', '.join(new_args)}):\n"
        code+= f"    return __{name}({', '.join(new_args)})\n"

        compiled_code = compile(code, "<string>", "exec")
        return types.FunctionType(compiled_code.co_consts[0], {f"__{name}": f}, name)

    def write_vti(self, fields, filename):
        if isinstance(fields, Function):
            fields = [fields]

        n = np.array(self.mesh.n + tuple([1] * (3 - self.mesh.dim))) + 1
        grid = pv.UniformGrid(dimensions = n, spacing = self.mesh.dx, origin=self.mesh.origin)

        for field in fields:
            if isinstance(field, str):
                name = field
                field = self.getattr(name)
            else:
                name = field.name

            if field.shape == ():
                if self.mesh.dim == 2:
                    data = field.tensor.detach().unsqueeze(-2).expand(-1,-1,2).cpu().numpy().flatten("F")
                elif self.mesh.dim == 3:
                    data = field.tensor.detach().cpu().numpy().flatten("F")
                else:
                    raise
            elif field.shape == (3,):
                if self.mesh.dim == 2:
                    data = field.tensor.detach().unsqueeze(-2).expand(-1,-1,2,-1).cpu().numpy().reshape(-1, 3, order="F")
                elif self.mesh.dim == 3:
                    data = field.tensor.detach().cpu().numpy().reshape(-1, 3, order="F")
                else:
                    raise
            else:
                raise NotImplemented('Unsupported shape.')

            if field.ftype == 'node':
                grid.point_data.set_array(data, name)
            elif field.ftype == 'cell':
                grid.cell_data.set_array(data, name)
            else:
                raise NotImplemented('Unsupported ftype.')

        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        grid.save(filename)

    def read_vti(self, filename, name = None):
        # TODO enable reading of 2D data
        fields = {}
        data = pv.read(filename)

        assert np.array_equal(self.mesh.n, np.array(data.dimensions) - 1)

        if name is None:
            name = data.array_names[0]

        if name in data.point_data.keys():
            ftype = 'node'
            n = tuple([x + 1 for x in self.mesh.n])
        elif name in data.cell_data.keys():
            ftype = 'cell'
            n = self.mesh.n
        else:
            raise

        vals = data.get_array(name)
        if len(vals.shape) == 1:
            dim = n
            shape = ()
        else:
            dim = n + (vals.shape[-1],)
            shape = (3,)

        return Function(self, ftype = ftype, shape = shape, tensor = self.tensor(vals.reshape(dim, order="F")))

    def domains_from_file(self, filename, scale = 1.):
        mesh = self.mesh

        # read image data and volume domains
        unstructured_mesh = pv.read(filename)

        # interpolate on mesh
        x = np.arange(mesh.n[0]) * mesh.dx[0] + mesh.dx[0]/2. + mesh.origin[0]
        y = np.arange(mesh.n[1]) * mesh.dx[1] + mesh.dx[1]/2. + mesh.origin[1]
        z = np.arange(mesh.n[2]) * mesh.dx[2] + mesh.dx[2]/2. + mesh.origin[2]
        points = np.stack(np.meshgrid(x, y, z, indexing = "ij"), axis=-1).reshape(-1,3) / scale

        containing_cells = unstructured_mesh.find_containing_cell(points)
        data = unstructured_mesh.get_array(0)[containing_cells]
        data[containing_cells == -1] = -1 # containing_cell == -1, if point is not included in any cell

        return Function(self, ftype = 'cell', tensor = self.tensor(data.reshape(mesh.n)))
