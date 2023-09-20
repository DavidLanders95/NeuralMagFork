import os
import numpy as np
import torch
import inspect
import types

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
        self._attr_values['mesh__dx'] = self.tensor(mesh.dx)
        self.t = 0.

        logging.info_green(f"[State] Running on device: {self._device}")
        logging.info_green("[Mesh] %dx%dx%d (size = %g x %g x %g)" % (mesh.n + mesh.dx))

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

    def tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(value, device = self.device, dtype = self.dtype) 

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
            container = self
            while '__' in arg:
                parent, child = arg.split('__', 1)
                container = getattr(container, parent)
                arg = child
            attr = getattr(container, arg)
            if hasattr(attr, 'tensor'):
                args.append(attr.tensor)
            else:
                args.append(attr)

        return func, args
