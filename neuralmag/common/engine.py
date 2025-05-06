# SPDX-License-Identifier: MIT

import json
import re
from functools import reduce
from itertools import product

import sympy as sp
import sympy.vector as sv
from scipy.special import p_roots
from tqdm import tqdm

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

        ## try to reduce multiplications of fields for better performance
        ## the switch on n_gaus is purely heuristic and was introduced for the cubic anisotropy
        if n_gauss == 1:
            cmd = str(sp.collect(sp.factor_terms(iexpr), symbs))
        else:
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
