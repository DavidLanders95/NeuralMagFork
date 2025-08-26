# SPDX-License-Identifier: MIT

import inspect
import os
import types

import numpy as np
import pyvista as pv

from neuralmag.common import CellFunction, Function, config, logging
from neuralmag.common.code_class import CodeClass

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

    def __delattr__(self, name):
        # don't mess with protected attributes
        if name[0] == "_":
            super().__delattr__(name, value)
            return
        return delattr(self._state, "material__" + name)


class State(CodeClass):
    r"""
    This class carries all information of the spatial discretization, parameters
    and the current state of the simulation.

    :param mesh: The mesh for the simulation
    :type mesh: class:`Mesh`
    :param device: The device to be used, defaults to "cpu" (torch backend only)
    :type device: str, optional
    :param dtype: The dtype to be used, defaults to "float64" (torch backend only)
    :type dtype: str, optional
    """

    def __init__(self, mesh, device=None, dtype=None):
        self._attr_values = {}
        self._attr_types = {}
        self._attr_funcs = {}

        if device == None:
            self._device = config.device
        else:
            self._device = config.backend.device_for_state(device)

        if dtype == None:
            self._dtype = config.dtype
        else:
            self._dtype = config.backend.dtype_for_state(dtype)

        self._material = Material(self)
        self._mesh = mesh
        self.dx = self.tensor(mesh.dx)
        self.t = 0.0

        self._attr_values["eps"] = config.backend.eps(self.dtype)

        self.save_and_load_code(mesh.dim)

        # initialize domain management
        self.domain = lambda domains: domains > 0
        self.subdomain = lambda domains, domain_id: domains == domain_id
        self.rho = CellFunction(
            self, tensor=lambda domain: config.backend.np.where(domain, 1.0, self.eps)
        )

        self.domains = CellFunction(self, dtype=config.backend.integer).fill(
            1, expand=True
        )

        # TODO allow surface regions also for lower dimensions
        if mesh.dim == 3:
            self.rhoxy = Function(self, "ccn", tensor=self._code.rhoxy)
            self.rhoxz = Function(self, "cnc", tensor=self._code.rhoxz)
            self.rhoyz = Function(self, "ncc", tensor=self._code.rhoyz)

        logging.info_green(
            f"[State] Running on device: {self.dx.device} (dtype = {self.dx.dtype}, backend = {config.backend_name})"
        )

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
        return self._dtype

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

    def tensor(self, value, **kwargs):
        """
        Creates a PyTorch tensor with device and dtype set according to the
        state defaults.

        :param value: The value of the tensor
        :type value: config.backend.Tensor, list
        :return: The tensor
        :rtype: :class:`config.backend.Tensor`
        """
        default_options = {"device": self.device, "dtype": self.dtype}
        options = {
            **default_options,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        return config.backend.tensor(value, **options)

    def zeros(self, shape, **kwargs):
        """
        Creates an empty tensor of given shape with default dtype on the default device.

        :param shape: The shape of the tensor
        :type shape: tuple
        :param **kwargs: Parameters passed to the PyTorch routine
        :return: The tensor
        :rtype: config.backend.Tensor
        """
        return config.backend.zeros(
            shape, device=self.device, dtype=self.dtype, **kwargs
        )

    def __getattr__(self, name):
        if callable(self._attr_values[name]):
            if not name in self._attr_funcs:
                attr = self._attr_values[name]
                self._attr_funcs[name] = self.resolve(attr)

            func = self._attr_funcs[name]
            args = []
            for arg in list(inspect.signature(func).parameters.keys()):
                attr = getattr(self, arg)
                if hasattr(attr, "tensor"):
                    args.append(attr.tensor)
                else:
                    args.append(attr)

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
            value = self.tensor(value)

        if isinstance(value, list):
            try:
                value = self.tensor(value)
            except ValueError:
                pass

        if callable(value):
            self._attr_funcs.clear()

        self._attr_values[name] = value

    def __delattr__(self, name):
        # don't mess with protected attributes
        if name[0] == "_":
            super().__delattr__(name)
            return

        attr = self._attr_values.pop(name, None)

        if callable(attr):
            self._attr_funcs.clear()

    def _collect_func_deps(self, attr, exclude=None, remap={}, inject={}):
        exclude = exclude or []
        func_names = []
        args = {}
        for arg in set(
            [remap.get(a, a) for a in inspect.signature(attr).parameters.keys()]
        ) - set(exclude):
            if arg in inject:
                attr = inject[arg]
            else:
                attr = self._attr_values[arg]

            if isinstance(attr, Function) and attr.func:
                attr = attr.func

            if callable(attr):
                func_names.append(arg)
                subfunc_names, subargs = self._collect_func_deps(
                    attr, exclude=exclude, remap=remap, inject=inject
                )
                func_names = [
                    f for f in func_names if f not in subfunc_names
                ] + subfunc_names
                args.update(subargs)
            else:
                args[arg] = attr

        return func_names, args

    def resolve(self, func, func_args=None, remap={}, inject={}):
        """
        Analyse arguments of supplied function and create Python function that
        depends solely on func_args if provided. If func_args is None the returned
        function will depend on all static state attributes that are dependencies of func.

        :param f: The function to by analyzed, if string is provided the state
                  attribute with the respective name is used.
        :type f: Callable, str
        :param func_args: Arguments of the returned function. If not set,
                          function takes all dependent attributes.
        :type func_args: list, optional
        :param remap: applies mapping to function and all dependent subfunctions
        :type remap: dict
        :param inject: callables to be injected instead of named attributes
        :type inject: dict
        :return: New function that takes args as arguments, if args is None
                 the functions has all dependencies as arguments.
        :rtype: tuple
        """
        if isinstance(func, str):
            func = self._attr_values[func]

        if isinstance(func, Function) and func.func:
            func = func.func

        func = self.remap(func, remap)

        subfunc_names, args = self._collect_func_deps(
            func, exclude=func_args, remap=remap, inject=inject
        )
        name = func.__name__
        name = "lmda" if func.__name__ == "<lambda>" else name

        # setup function with all dependencies
        if subfunc_names or func_args is not None:
            code = (
                f"def {name}({', '.join(args if func_args is None else func_args)}):\n"
            )
            globals = {}
            for subfunc_name in reversed(subfunc_names):
                if subfunc_name in inject:
                    subfunc = inject[subfunc_name]
                else:
                    subfunc = self._attr_values[subfunc_name]
                    if isinstance(subfunc, Function) and subfunc.func:
                        subfunc = subfunc.func
                globals[f"__{subfunc_name}"] = subfunc
                code += (
                    f"    {subfunc_name} ="
                    f" __{subfunc_name}({', '.join([remap.get(a, a) for a in inspect.signature(subfunc).parameters.keys()])})\n"
                )
            globals[f"__{name}"] = func
            code += (
                "    return"
                f" __{name}({', '.join(list(inspect.signature(func).parameters.keys()))})\n"
            )

            # populate globals with bound variables
            if func_args is not None:
                for arg in list(set(args) - set(func_args)):
                    attr = getattr(self, arg)
                    if hasattr(attr, "tensor"):
                        globals[arg] = attr.tensor
                    else:
                        globals[arg] = attr

            compiled_code = compile(code, "<string>", "exec")
            return types.FunctionType(compiled_code.co_consts[0], globals, name)
        else:
            return func

    @staticmethod
    def remap(f, mapping):
        """
        Remaps the arguments of a given function according the the provided mapping.

        :param f: The function to be remapped
        :type f: Callable
        :param mapping: The name mapping of the arguments
        :type mapping: dict
        :return: The remapped function
        :rtype: Callable

        :Example:
            .. code-block::

                state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))

                def f(a, b):
                   return a + b

                g = state.remap(f, {"a": "x", "b": "y"})
                # g is a function with arguments "x" and "y"
        """
        if mapping == {}:
            return f

        name = "lmda" if f.__name__ == "<lambda>" else f.__name__
        old_args = list(inspect.signature(f).parameters.keys())
        new_args = [mapping.get(a, a) for a in old_args]

        if old_args == new_args:
            return f

        code = f"def {name}({', '.join(new_args)}):\n"
        code += f"    return __{name}({', '.join(new_args)})\n"

        compiled_code = compile(code, "<string>", "exec")
        return types.FunctionType(compiled_code.co_consts[0], {f"__{name}": f}, name)

    def coordinates(self, spaces=None, numpy=False):
        """
        Returns 3 tensors containing the x, y, z coordinates of each cell/node
        of the mesh. In the case of cell discretization the coordinates at the
        cell centers are provided. In the case of node discretization the node
        positions are returned.

        :param spaces: function spaces, e.g. "ccc", "nnn"
        :type spaces: str
        :param numpy: return numpy arrays instead of backend arrays
        :type numpy: bool
        :return: The coordinates
        :rtype: config.backend.Tensor

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
                    config.backend.linspace(
                        self.dx[i] / 2.0 + self.mesh.origin[i],
                        self.dx[i] / 2.0
                        + self.mesh.origin[i]
                        + self.dx[i] * (self.mesh.n[i] - 1.0),
                        self.mesh.n[i],
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            elif space == "n":
                ranges.append(
                    config.backend.linspace(
                        self.mesh.origin[i],
                        self.mesh.origin[i] + self.dx[i] * self.mesh.n[i],
                        self.mesh.n[i] + 1,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            else:
                raise NotImplementedError(f"Unknown function space '{space}'.")

        coords = config.backend.meshgrid(*ranges, indexing="ij")
        if numpy:
            return tuple(config.backend.to_numpy(c) for c in coords)
        else:
            return coords

    def add_domain(self, id, condition):
        # initialize full tensor zero
        if (self.domains.tensor == 1).all():
            self.domains.fill(0)
        self.domains.tensor = config.backend.np.where(
            condition, id, self.domains.tensor
        )

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

            # check for spatial dimension and pure cell/node data
            if len(field.spaces) > 3:
                raise AttributeError(
                    "VTI only supports spatial dimensions smaller or equal than 3"
                )
            if len(set(field.spaces)) > 1:
                raise AttributeError(
                    "VTI only supports pure cell/nodal function spaces"
                )
            else:
                space = field.spaces[0]

            data = config.backend.to_numpy(field.tensor)

            # extend data to length 2 in hidden dimensions in case of nodal discretization
            if space == "n":
                missing_dims = tuple(
                    np.arange(3 - len(field.spaces)) + len(field.spaces)
                )
                data = np.expand_dims(data, missing_dims)
                new_shape = np.array(data.shape)
                new_shape[
                    missing_dims,
                ] = 2
                data = np.broadcast_to(data, new_shape)

            if field.shape == ():
                data = data.flatten("F")
            elif field.shape == (3,):
                data = data.reshape(-1, 3, order="F")
            else:
                raise NotImplemented(f"Unsupported shape '{field.shape}'.")

            if space == "n":
                grid.point_data.set_array(data, name)
            elif space == "c":
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

        assert np.array_equal(
            self.mesh.n + (1,) * (3 - self.mesh.dim), np.array(data.dimensions) - 1
        )

        if name is None:
            name = data.array_names[0]

        n = self.mesh.n + (1,) * (3 - self.mesh.dim)
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
        if self.mesh.dim == 1:
            values = values[:, 0, 0, ...]
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

    @classmethod
    def _generate_code(cls, dim):
        if dim < 3:
            return

        code = config.backend.CodeBlock()

        # generate interface rho attributes
        with code.add_function("rhoxy", ["rho"]) as func:
            func.zeros_like(
                "rhoxy1", "rho", shape="(rho.shape[0], rho.shape[1], rho.shape[2]+1)"
            )
            func.zeros_like(
                "rhoxy2", "rho", shape="(rho.shape[0], rho.shape[1], rho.shape[2]+1)"
            )
            func.add_to("rhoxy1", ":,:,:-1", "rho")
            func.add_to("rhoxy2", ":,:,1:", "rho")
            func.retrn_maximum("rhoxy1", "rhoxy2")

        with code.add_function("rhoxz", ["rho"]) as func:
            func.zeros_like(
                "rhoxz1", "rho", shape="(rho.shape[0], rho.shape[1]+1, rho.shape[2])"
            )
            func.zeros_like(
                "rhoxz2", "rho", shape="(rho.shape[0], rho.shape[1]+1, rho.shape[2])"
            )
            func.add_to("rhoxz1", ":,:-1,:", "rho")
            func.add_to("rhoxz2", ":,1:,:", "rho")
            func.retrn_maximum("rhoxz1", "rhoxz2")

        with code.add_function("rhoyz", ["rho"]) as func:
            func.zeros_like(
                "rhoyz1", "rho", shape="(rho.shape[0]+1, rho.shape[1], rho.shape[2])"
            )
            func.zeros_like(
                "rhoyz2", "rho", shape="(rho.shape[0]+1, rho.shape[1], rho.shape[2])"
            )
            func.add_to("rhoyz1", ":-1,:,:", "rho")
            func.add_to("rhoyz2", "1:,:,:", "rho")
            func.retrn_maximum("rhoyz1", "rhoyz2")

        return code
