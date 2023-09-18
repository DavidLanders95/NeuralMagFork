import os
import numpy as np
import torch
import inspect
import types

from . import logging

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
    def __init__(self, mesh, t0=0.0, device=None):
        self._attr_values = {}
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
        self.t = t0

        logging.info_green("[State] running on device:%s" % self._device)
        logging.info_green("[Mesh] %dx%dx%d (size= %g x %g x %g)" % (mesh.n + mesh.dx))

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return torch.float64

    @property
    def t(self):
        return self._t

    @property
    def material(self):
        return self._material

    def __getattr__(self, name):
        if callable(self._attr_values[name]):
            func, args = self.get_func(name)
            return func(*args)
        else:
            return self._attr_values[name]

    def __setattr__(self, name, value): 
        # don't mess with protected attributes
        if name[0] == '_':
            super().__setattr__(name, value) 
            return  

        if name == 't':
            if hasattr(self._attr_values, 't'):
                self._attr_values['t'].fill_(t)
            else:
                value = torch.tensor(value, dtype = self.dtype, device = self.device)

        # TODO check for __ in name (material)
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

    def get_func(self, name):
        if not hasattr(self._attr_funcs, name):
            attr = self._attr_values[name]
            func_names, args = self._collect_func_deps(attr)

            # setup function with all dependencies
            if func_names:
                code = f"def {name}({', '.join(sorted(args))}):\n"
                func_pointers= {}
                for func_name in reversed(func_names):
                    func = self._attr_values[func_name]
                    func_pointers[f"__{func_name}"] = func
                    code += f"    {func_name} = __{func_name}({', '.join(list(inspect.signature(func).parameters.keys()))})\n"
                func_pointers[f"__{name}"] = attr
                code += f"    return __{name}({', '.join(list(inspect.signature(attr).parameters.keys()))})\n"

                compiled_code = compile(code, "<string>", "exec")
                func = types.FunctionType(compiled_code.co_consts[0], func_pointers, name)
            else:
                func = attr

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

            self._attr_funcs[name] = (func, args)

        return self._attr_funcs[name]
