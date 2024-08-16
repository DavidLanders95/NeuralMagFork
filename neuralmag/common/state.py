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

import inspect
import os
import types

import numpy as np
import pyvista as pv
import torch

from . import CellFunction, Function, logging

__all__ = ["State"]


class Material:
    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        return getattr(self._state, "material__" + name)

    def __setattr__(self, name, value):
        # don't mess with protected attributes
        if name[0] == "_":
            super().__setattr__(name, value)
            return
        return setattr(self._state, "material__" + name, value)


class State(object):
    r"""
    This class carries all information of the spatial discretization, parameters
    and the current state of the simulation.

    :param mesh: The mesh for the simulation
    :type mesh: class:`Mesh`
    :param device: The PyTorch device to be used, defaults to "cpu"
    :type device: str, optional
    """

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
        self.t = 0.0

        # initialize density fields for volume and facet measure with 1
        self.rho = CellFunction(self).fill(1.0, expand=True)
        if mesh.dim == 3:
            self.rhoxy = Function(self, "ccn").fill(1.0, expand=True)
            self.rhoxz = Function(self, "cnc").fill(1.0, expand=True)
            self.rhoyz = Function(self, "ncc").fill(1.0, expand=True)

        self._attr_values["eps"] = torch.finfo(self.dtype).eps
        logging.info_green(f"[State] Running on device: {self._device}")

    @property
    def device(self):
        """
        The PyTorch device used for all tensors.
        """
        return self._device

    @property
    def dtype(self):
        """
        The PyTorch dtype used for all tensors.
        """
        return torch.float64

    @property
    def mesh(self):
        """
        The mesh
        """
        return self._mesh

    @property
    def material(self):
        """
        The material namespace.

        The namespace supports the same functionality
        as the :class:`State` class to set and get regular and dynamic attributes.

        :Example:
            .. code-block::

                 mesh = nm.Mesh((10, 10, 1), (5e-9, 5e-9, 3e-9))
                 state = nm.State(mesh)

                 # Set saturation magnetization Ms according to Bloch's law
                 Ms0 = 8e5
                 Tc = 400.0
                 state.T = 200.0
                 state.material.Ms = lambda T: Ms0 * (1 - T/Tc)**1.5
        """
        return self._material

    def getattr(self, name):
        """
        Returns the attribute for the given name. Attributes in namespaces
        can be accessed by using "." as a seperator, e.g. :code:`material.Ms`.

        :param name: The name of the attribute
        :type name: str
        :return: The value of the attribute
        """
        container = self
        while "." in name:
            parent, child = name.split(".", 1)
            container = getattr(container, parent)
            name = child
        return getattr(container, name)

    def tensor(self, value, requires_grad=False):
        """
        Creates a PyTorch tensor with device and dtype set according to the
        state defaults.

        :param value: The value of the tensor
        :type value: torch.Tensor, list
        :param requires_grad: Flag to indicate if tensor should be tracked for autodifferentiation
        :type requires_grad: bool
        :return: The tensor
        :rtype: :class:`torch.Tensor`
        """
        if isinstance(value, torch.Tensor):
            if value.device != self.device:
                return value.to(self.device)
            else:
                return value
        return torch.tensor(
            value, device=self.device, dtype=self.dtype, requires_grad=requires_grad
        )

    def zeros(self, shape, **kwargs):
        """
        Creates an empty tensor of given shape with default dtype on the default device.

        :param shape: The shape of the tensor
        :type shape: tuple
        :param \**kwargs: Parameters passed to the PyTorch routine
        :return: The tensor
        :rtype: torch.Tensor
        """
        return torch.zeros(shape, device=self.device, dtype=self.dtype, **kwargs)

    def __getattr__(self, name):
        if callable(self._attr_values[name]):
            if not name in self._attr_funcs:
                attr = self._attr_values[name]
                self._attr_funcs[name] = self.get_func(attr)
            func, args = self._attr_funcs[name]
            value = func(*args)

        else:
            value = self._attr_values[name]

        if name in self._attr_types:
            spaces, shape = self._attr_types[name]
            return Function(self, spaces=spaces, shape=shape, tensor=value)
        else:
            return value

    def __setattr__(self, name, value):
        # don't mess with protected attributes
        if name[0] == "_":
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

        if isinstance(value, list):
            try:
                value = self.tensor(value)
            except ValueError:
                pass

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
                func_names = [
                    f for f in func_names if f not in subfunc_names
                ] + subfunc_names
                args.update(subargs)
            else:
                args[arg] = attr

        return func_names, args

    def get_func(self, f, add_args=None):
        """
        Analyse arguments of supplied function and create Python function that
        depends solely on static state attributes of the state.

        :param f: The function to by analyzed
        :type f: Callable
        :param add_args: Additional arguments to be added. Arguments provided
                         here are always used as the first arguments in the
                         signature.
        :type add_args: list, optional
        :return: New function that only depends on static state attributes and
                 list of references to the static attributes.
        :rtype: tuple
        """
        add_args = add_args or []
        func_names, args = self._collect_func_deps(f)
        args = list(set(args) - set(add_args))
        name = f.__name__
        name = "lmda" if f.__name__ == "<lambda>" else name

        # setup function with all dependencies
        if func_names:
            code = f"def {name}({', '.join(add_args + sorted(args))}):\n"
            func_pointers = {}
            for func_name in reversed(func_names):
                func = self._attr_values[func_name]
                func_pointers[f"__{func_name}"] = func
                code += (
                    f"    {func_name} ="
                    f" __{func_name}({', '.join(list(inspect.signature(func).parameters.keys()))})\n"
                )
            func_pointers[f"__{name}"] = f
            code += (
                "    return"
                f" __{name}({', '.join(list(inspect.signature(f).parameters.keys()))})\n"
            )
            compiled_code = compile(code, "<string>", "exec")
            func = types.FunctionType(compiled_code.co_consts[0], func_pointers, name)
        else:
            func = f

        # collect args
        args = []
        for arg in list(inspect.signature(func).parameters.keys()):
            attr = self.getattr(arg.replace("__", "."))
            if hasattr(attr, "tensor"):
                args.append(attr.tensor)
            else:
                args.append(attr)

        return func, args

    @staticmethod
    def wrap_func(f, mapping):
        """
        Wrappes a given function into another function renaming its arguments
        according to the mapping provided.

        :param f: The function to be wrapped
        :type f: Callable
        :param mapping: The name mapping of the arguments
        :type mapping: dict
        :return: The wrapped function
        :rtype: Callable

        :Example:
            .. code-block::

                state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))

                def f(a, b):
                   return a + b

                g = state.wrap_func(f, {"a": "x", "b": "y"})
                # g is a function with arguments "x" and "y"
        """
        name = "lmda" if f.__name__ == "<lambda>" else f.__name__
        old_args = list(inspect.signature(f).parameters.keys())
        new_args = [mapping.get(a, a) for a in old_args]

        code = f"def {name}({', '.join(new_args)}):\n"
        code += f"    return __{name}({', '.join(new_args)})\n"

        compiled_code = compile(code, "<string>", "exec")
        return types.FunctionType(compiled_code.co_consts[0], {f"__{name}": f}, name)

    def coordinates(self, spaces=None):
        """
        Returns 3 tensors containing the x, y, z coordinates of each cell/node
        of the mesh. In the case of cell discretization the coordinates at the
        cell centers are provided. In the case of node discretization the node
        positions are returned.

        :param spaces: function spaces, e.g. "ccc", "nnn"
        :type spaces: str
        :return: The coordinates
        :rtype: torch.Tensor

        :Example:
            .. code-block::

                state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
                x, y, z = state.coordinates('nnn')

                # initialize magnetization based on coordinate function
                state.m = VectorFunction(state)
                state.m.tensor[..., 0] = torch.sin(x/20e-9)
                state.m.tensor[..., 1] = torch.cos(x/20e-9)
        """
        if spaces == None:
            spaces = "c" * self.mesh.dim

        ranges = []
        for i, space in enumerate(spaces):
            if space == "c":
                ranges.append(
                    torch.arange(
                        self.dx[i] / 2.0 + self.mesh.origin[i],
                        self.dx[i] * self.mesh.n[i] + self.mesh.origin[i],
                        self.dx[i],
                        dtype=self.dtype,
                    )
                )
            elif space == "n":
                ranges.append(
                    torch.arange(
                        self.mesh.origin[i],
                        self.dx[i] * self.mesh.n[i] + self.mesh.origin[i] + self.eps,
                        self.dx[i],
                        dtype=self.dtype,
                    )
                )
            else:
                raise NotImplementedError(f"Unknown function space '{space}'.")

        return torch.meshgrid(*ranges, indexing="ij")

    def write_vti(self, fields, filename):
        """
        Write field data into VTI file.

        :param fields: The field data to be written. The field can be provided
            either as a :class:`Function` or as attribute name(s) of state
            attributes
        :type fields: str, Function, list
        :param filename: The name of the VTI file
        :type filename: str

        :Example:
            .. code-block::

                state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))

                state.material.Ms = CellFunction(state).fill(8e5)
                state.m = VectorFunction(state).fill([0, 0, 1])

                # Write m into m.vti
                state.write_vti("m", "m.vti")

                # Write Ms and m into data.vti
                state.write_vti(["material.Ms", "m"], "data.vti")

                # Write some function f into f.vti
                f = Function(state)
                state.write_vti(f, "f.vti")
        """
        if isinstance(fields, (Function, str)):
            fields = [fields]

        n = np.array(self.mesh.n + tuple([1] * (3 - self.mesh.dim))) + 1
        grid = pv.ImageData(dimensions=n, spacing=self.mesh.dx, origin=self.mesh.origin)

        for field in fields:
            if isinstance(field, str):
                name = field
                field = self.getattr(name)
            else:
                name = field.name

            if field.shape == ():
                if field.spaces == "nn":
                    data = (
                        field.tensor.detach()
                        .unsqueeze(-1)
                        .expand(-1, -1, 2)
                        .cpu()
                        .numpy()
                        .flatten("F")
                    )
                else:
                    data = field.tensor.detach().cpu().numpy().flatten("F")
            elif field.shape == (3,):
                if field.spaces == "nn":
                    data = (
                        field.tensor.detach()
                        .unsqueeze(-2)
                        .expand(-1, -1, 2, -1)
                        .cpu()
                        .numpy()
                        .reshape(-1, 3, order="F")
                    )
                else:
                    data = field.tensor.detach().cpu().numpy().reshape(-1, 3, order="F")
            else:
                raise NotImplemented(f"Unsupported shape '{field.shape}'.")

            if field.spaces == "nn" or field.spaces == "nnn":
                grid.point_data.set_array(data, name)
            elif field.spaces == "cc" or field.spaces == "ccc":
                grid.cell_data.set_array(data, name)
            else:
                raise NotImplemented(f"Unsupported space '{field.spaces}'.")

        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        grid.save(filename)

    def read_vti(self, filename, name=None):
        """
        Read field data from VTI file.

        :param filename: The filename of the VTI file
        :type filename: str
        :param name: The name of the attribute in the VTI file. If not
            provided, the first field in the VTI file will be read.
        :type name: str, None
        :return: The function
        :rvalue: :class:`Function`
        """
        fields = {}
        data = pv.read(filename)

        if self.mesh.dim == 2:
            assert np.array_equal(self.mesh.n + (1,), np.array(data.dimensions) - 1)
        else:
            assert np.array_equal(self.mesh.n, np.array(data.dimensions) - 1)

        if name is None:
            name = data.array_names[0]

        n = self.mesh.n
        if self.mesh.dim == 2:
            n += (1,)
        if name in data.point_data.keys():
            spaces = "n" * self.mesh.dim
            n = tuple([x + 1 for x in n])
        elif name in data.cell_data.keys():
            spaces = "c" * self.mesh.dim
        else:
            raise RuntimeError(f"Field '{name}' not found in VTI file.")

        vals = data.get_array(name)
        if len(vals.shape) == 1:
            dim = n
            shape = ()
        else:
            dim = n + (vals.shape[-1],)
            shape = (3,)

        values = self.tensor(vals.reshape(dim, order="F"))
        if self.mesh.dim == 2:
            values = values[:, :, 0, ...]

        return Function(self, spaces=spaces, shape=shape, tensor=values)

    def domains_from_file(self, filename, scale=1.0):
        mesh = self.mesh

        # read image data and volume domains
        unstructured_mesh = pv.read(filename)

        # interpolate on mesh
        x = np.arange(mesh.n[0]) * mesh.dx[0] + mesh.dx[0] / 2.0 + mesh.origin[0]
        y = np.arange(mesh.n[1]) * mesh.dx[1] + mesh.dx[1] / 2.0 + mesh.origin[1]
        z = np.arange(mesh.n[2]) * mesh.dx[2] + mesh.dx[2] / 2.0 + mesh.origin[2]
        points = (
            np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
            / scale
        )

        containing_cells = unstructured_mesh.find_containing_cell(points)
        data = unstructured_mesh.get_array(0)[containing_cells]
        data[
            containing_cells == -1
        ] = -1  # containing_cell == -1, if point is not included in any cell

        return Function(
            self, spaces="c" * mesh.dim, tensor=self.tensor(data.reshape(mesh.n))
        )
