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

import hashlib
import importlib
import json
import os
import pathlib
import pickle
import re
from functools import reduce
from itertools import product

import sympy as sp
import sympy.vector as sv
import torch
from scipy import constants
from scipy.special import p_roots
from tqdm import tqdm

from ..common import config, logging

N = sv.CoordSys3D("N")
cs_dx = sp.symbols("_dx[0]_ _dx[1]_ _dx[2]_", real=True, positive=True)
cs_x = [N.x, N.y, N.z]
cs_e = [N.i, N.j, N.k]


def dX(**kwargs):
    r"""
    Generic integration measure

    :param \**kwargs: Arbitrary meta information
    :return: The integration measure
    :rtype: sympy.Expr
    """
    # TODO implement singleton
    return sp.Symbol(f"_dX:{json.dumps(kwargs)}_")


def dV(dim=3, region="rho", **kwargs):
    r"""
    Volume integral measure

    :param dim: Dimension of the mesh to integrate over.
    :type dim: int
    :param region: name of the cell function that acts as a region indicator
    :type region: str
    :param \**kwargs: Additional meta information
    :return: The integration measure
    :rtype: sympy.Expr
    """
    rho = Variable(region, "c" * dim)
    return rho * dX(dims=[None, None, None], **kwargs)


def dA(dim=3, normal=2, region="rhoxy", idx=":", **kwargs):
    r"""
    Volume integral measure

    :param dim: Dimension of the mesh to integrate over.
    :type dim: int
    :param region: name of the cell function that acts as a region indicator
    :type region: str
    :param \**kwargs: Additional meta information
    :return: The integration measure
    :type: sympy.Expr
    """
    assert dim == 3
    spaces = ["c"] * 3
    spaces[normal] = "n"
    rho = Variable(region, "".join(spaces))
    dims = [None, None, None]
    dims[normal] = idx
    return rho * dX(dims=dims, **kwargs)


def compile(func):
    if config.torch["compile"]:
        return torch.compile(func)
    else:
        return func


class CodeFunction(object):
    def __init__(self, block, name, variables):
        self._block = block
        self._name = name
        self._variables = variables

    def __enter__(self):
        self._code = f"def {self._name}({', '.join(self._variables)}):\n"
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return False
        self._block.add(self._code)
        self._block.add("\n")
        return True

    @staticmethod
    def sum(*terms):
        return " + ".join([f"({term}).sum()" for term in terms])

    def add_line(self, code):
        self._code += f"    {code}\n"

    def assign(self, lhs, rhs):
        self.add_line(f"{lhs} = {rhs}")

    def assign_sum(self, lhs, *terms):
        self.assign(lhs, self.sum(*terms))

    def zeros_like(self, var, src, shape=None):
        if shape is None:
            self.add_line(f"{var} = torch.zeros_like({src})")
        else:
            self.add_line(
                f"{var} = torch.zeros({shape}, dtype = {src}.dtype, device ="
                f" {src}.device)"
            )

    def add_to(self, var, idx, rhs):
        self.add_line(f"{var}[{idx}] += {rhs}")

    def retrn(self, code):
        self.add_line(f"return {code}")

    def retrn_sum(self, *terms):
        self.add_line(f"return {self.sum(*terms)}")


class CodeBlock(object):
    def __init__(self):
        self._code = "import torch\n\n"

    def add_function(self, name, variables):
        return CodeFunction(self, name, variables)

    def add(self, code):
        self._code += code

    def __str__(self):
        return self._code


class CodeClass(object):
    def save_and_load_code(self, *args):
        # setup cache file name
        this_module = pathlib.Path(importlib.import_module(self.__module__).__file__)
        i = this_module.parent.parts[::-1].index("neuralmag")
        prefix = "_".join(this_module.parent.parts[-i:] + (this_module.stem,))
        cache_file = f"{prefix}_{hashlib.md5(pickle.dumps(args)).hexdigest()}.py"
        cache_dir = os.getenv(
            "NEURALMAG_CACHE", pathlib.Path.home() / ".cache" / "neuralmag"
        )
        code_file_path = cache_dir / cache_file

        # generate code
        if not code_file_path.is_file():
            code_file_path.parent.mkdir(parents=True, exist_ok=True)
            # TODO check if _generate_code method exists
            logging.info_green(
                f"[{self.__class__.__name__}] Generate torch core methods"
            )
            code = str(self._generate_code(*args))
            with open(code_file_path, "w") as f:
                f.write(code)

        # import code
        module_spec = importlib.util.spec_from_file_location("code", code_file_path)
        self._code = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self._code)


def Variable(name, spaces, shape=()):
    r"""
    Symbolic representation of a field given as SymPy expression.

    :param name: The name of the field
    :type name: str
    :param spaces: The function spaces of the field in the principal coordinate
                   directions given as string with 'c' representing a cell-based
                   discretization and 'n' representing a node-based discretization.
    :type spaces: str
    :param shape: The shape (dimension) of the field, e.g. () for a scalar field and
                  (3,) for a vector field
    :type shape: tuple
    :return: The variable
    :rtype: sympy.Expr
    """
    result = []
    for idx in product(*[{"n": [0, 1], "c": [None]}[s] for s in spaces]):
        phi = 1.0
        for i, j in enumerate(idx):
            if j is not None:
                phi *= 1 - cs_x[i] / cs_dx[i] + 2 * j * cs_x[i] / cs_dx[i] - j
        if shape == ():
            result.append(
                sp.Symbol(f"_{name}:{spaces}:{shape}:{list(idx)}_", real=True) * phi
            )
        elif shape == (3,):
            for l in range(3):
                result.append(
                    sp.Symbol(f"_{name}:{spaces}:{shape}:{list(idx) + [l]}_", real=True)
                    * phi
                    * cs_e[l]
                )
        else:
            raise Exception("Shape not supported")
    return reduce(lambda x, y: x + y, result)


def integrate(expr, dims, n=3):
    x, w = p_roots(n)

    integrand = expr
    for i, dim in enumerate(dims):
        if dim is None:
            integral = 0
            for j in range(n):
                integral += (
                    w[j]
                    * cs_dx[i]
                    / 2
                    * integrand.subs(cs_x[i], (1 + x[j]) * cs_dx[i] / 2)
                )
        else:
            integral = integrand.subs(cs_x[i], 0.0)
        integrand = integral

    return integral


def compile_functional(expr, n_gauss=3):
    # extract all integral measures with parameters and check consistency
    measure_symbols = [s for s in expr.free_symbols if re.match(r"^_dX:(.*)_$", s.name)]
    integrals = sp.collect(expr, measure_symbols, exact=True, evaluate=False)
    assert 1 not in integrals

    cmds = []
    variables = {"dx"}
    for symb in measure_symbols:
        match = re.match(r"^_dX:(.*)_$", symb.name)
        args = json.loads(match[1])

        # integrate
        # TODO use | operator for python 3.9
        iexpr = integrate(integrals[symb], **{**{"n": n_gauss}, **args})

        # skip zero integrals
        if iexpr.is_zero:
            continue

        # find all named symbols (fields)
        symbs = [
            symb
            for symb in iexpr.free_symbols
            if re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
        ]

        if len(symbs) == 0:
            raise Exception("Need at least one variable to integrate.")

        # try to reduce multiplications of fields for better performance
        cmd = str(sp.collect(sp.factor_terms(sp.expand(iexpr)), symbs))

        # retrieve topological dimension from first symbol
        match = re.match(r"^_(.*:.*:.*:.*)_$", symbs[0].name)
        shape, idx = [eval(x) for x in match[1].split(":")[2:]]
        dim = len(idx) - len(shape)

        for symb in symbs:
            match = re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
            name, spaces = match[1].split(":")[:2]
            shape, idx = [eval(x) for x in match[1].split(":")[2:]]

            variables.add(name)

            sidx = []
            for i, space in enumerate(spaces):
                if space == "n":
                    if args["dims"][i] is None:
                        sidx.append([":-1", "1:"][idx[i]])
                    else:
                        sidx.append(str(args["dims"][i]))
                        if isinstance(args["dims"][i], str):
                            variables.add(args["dims"][i])
                elif space == "c":
                    if args["dims"][i] is None:
                        sidx.append(":")
                    else:
                        raise Exception("Use node discretization in normal direction.")

            if shape == (3,):
                sidx.append(str(idx[-1]))

            # contract leading sequence of ":,: to ...
            arr_idx = re.sub(r"^(:,)*:($|,)", r"...\2", ",".join(sidx))
            cmd = cmd.replace(symb.name, f"{name}[{arr_idx}]")

        args["cmd"] = re.sub(r"_(dx\[\d\])_", r"\1", cmd)
        cmds.append(args)

    return cmds, variables


def linear_form_cmds(expr, n_gauss=3):
    cmds = []
    v = {}

    # collect all test functions in expr
    for symb in sorted(list(expr.free_symbols), key=lambda s: s.name):
        match = re.match(r"^_v:(.*:.*:.*)_$", symb.name)
        if match:
            v[symb] = match[1].split(":")
            v[symb][1:] = [eval(x) for x in v[symb][1:]]

    # retrieve topological dimension from first symbol
    _, shape, idx = next(iter(v.values()))
    dim = len(idx) - len(shape)

    # process test functions
    variables = set()
    for vsymb in tqdm(v, desc="Generating..."):
        vexpr = expr.xreplace(dict([(s, 1.0) if s == vsymb else (s, 0.0) for s in v]))
        terms, vvars = compile_functional(vexpr, n_gauss)
        variables = variables.union(vvars)
        vspaces, vshape, vidx = v[vsymb]

        for term in terms:
            # TODO why call it term here and args in compile_function?
            sidx = []
            for i, space in enumerate(vspaces):
                if space == "n":
                    if term["dims"][i] is None:
                        sidx.append([":-1", "1:"][vidx[i]])
                    else:
                        sidx.append(str(term["dims"][i]))
                elif space == "c":
                    if term["dims"][i] is None:
                        sidx.append(":")
                    else:
                        raise Exception("Use node discretization in normal direction.")

            if shape == (3,):
                sidx.append(str(vidx[-1]))

            cmds.append((",".join(sidx), term["cmd"]))

    return cmds, variables


def gateaux_derivative(expr, var):
    r"""
    Compute the Gateaux derivative (variation) of a functional with respect to
    a given variable.

    :param expr: Functional to be derived
    :type expr: sympy.Expr
    :param var: The variable used for the derivative
    :type var: :class:`Variable`
    :return: The resulting linear form
    :rtype: sympy.Expr
    """
    result = []
    for symb in var.free_symbols:
        if not hasattr(symb, "name") or not re.match(r"^_(.*:.*:.*:.*)_$", symb.name):
            continue
        v = sp.Symbol(re.sub(r"^_.*:(.*:.*:.*_)$", r"_v:\1", symb.name))
        result.append(v * expr.diff(symb))
    return reduce(lambda x, y: x + y, result)


def linear_form_code(form, n_gauss=3):
    r"""
    Generate PyTorch function for the evaluation of a given linear form.

    :param form: The linear form
    :type form: sympy.Expr
    :param n_gauss: Degree of Gauss integration
    :type n_gauss: int
    :return: The Python code of the PyTorch function
    :rtype: str
    """
    cmds, variables = linear_form_cmds(form, n_gauss)
    code = CodeBlock()
    with code.add_function("L", ["result"] + sorted(list(variables))) as f:
        for cmd in cmds:
            f.add_to("result", cmd[0], cmd[1])

    return code


def functional_code(form, n_gauss=3):
    r"""
    Generate PyTorch function for the evaluation of a given functional form.

    :param form: The functional
    :type form: sympy.Expr
    :param n_gauss: Degree of Gauss integration
    :type n_gauss: int
    :return: The Python code of the PyTorch function
    :rtype: str
    """
    terms, variables = compile_functional(form, n_gauss)
    code = CodeBlock()
    with code.add_function("M", sorted(list(variables))) as f:
        f.retrn_sum(*[term["cmd"] for term in terms])

    return code
