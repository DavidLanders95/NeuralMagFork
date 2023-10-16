import re
from functools import reduce
from itertools import product

import sympy as sp
import sympy.vector as sv
import re
from scipy.special.orthogonal import p_roots
from scipy import constants
from itertools import product
from functools import reduce
from tqdm import tqdm
import pathlib
import importlib
import torch
from ..common import logging, config

dx, dy, dz = sp.symbols('_dx[0]_ _dx[1]_ _dx[2]_', real=True, positive=True)
N = sv.CoordSys3D('N')

def compile(func):
    if config.torch['compile']:
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

    def add_line(self, code):
        self._code += f"    {code}\n"

    def assign(self, lhs, rhs):
        self.add_line(f"{lhs} = {rhs}")

    def zeros_like(self, var, src, shape=None):
        if shape is None:
            self.add_line(f"{var} = torch.zeros_like({src})")
        else:
            self.add_line(
                f"{var} = torch.zeros({shape}, dtype = {src}.dtype, device = {src}.device)")

    def add_to(self, var, idx, rhs):
        self.add_line(f"{var}[{idx}] += {rhs}")

    def retrn(self, code):
        self.add_line(f"return {code}")

    def retrn_sum(self, code):
        self.add_line(f"return ({code}).sum()")


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
        this_module = pathlib.Path(importlib.import_module(self.__module__).__file__)
        code_file_path = this_module.parent / 'code' / f"{this_module.stem}_{hash(frozenset(args))}.py"

        # generate code
        if not code_file_path.is_file():
            code_file_path.parent.mkdir(parents=True, exist_ok=True)
            # TODO check if generate_code method exists
            logging.info_green(f"[{self.__class__.__name__}] Generate torch core methods")
            code = str(self.generate_code(*args))
            with open(code_file_path, 'w') as f:
                f.write(code)

        # import code
        module_spec = importlib.util.spec_from_file_location('code', code_file_path)
        self._code = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self._code)

def Variable(name, space, dim = 3, shape = ()):
    result = []
    if space == 'node':
        if dim == 2:
            for i,j in product([0,1],[0,1]):
                phi = (1 - N.x/dx + 2*i*N.x/dx - i) * \
                      (1 - N.y/dy + 2*j*N.y/dy - j)

                if shape == ():
                    result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j]}_", real=True) * phi)
                elif shape == (3,):
                    for l in range(3):
                        result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,l]}_", real=True) * phi * [N.i, N.j, N.k][l])
        elif dim == 3:
            for i,j,k in product([0,1],[0,1],[0,1]):
                phi = (1 - N.x/dx + 2*i*N.x/dx - i) * \
                      (1 - N.y/dy + 2*j*N.y/dy - j) * \
                      (1 - N.z/dz + 2*k*N.z/dz - k)

                if shape == ():
                    result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k]}_", real=True) * phi)
                elif shape == (3,):
                    for l in range(3):
                        result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k,l]}_", real=True) * phi * [N.i, N.j, N.k][l])
        else:
            raise

    elif space == 'cell':
        if shape == ():
            result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[0]*dim}_", real=True))
        elif shape == (3,):
            for l in range(3):
                result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[0]*dim + [l]}_", real=True) * [N.i, N.j, N.k][l])
    else:
        raise NotImplemented
    return reduce(lambda x, y: x + y, result)


def integrate(expr, n=3):
    x, w = p_roots(n)
    intx = 0
    for i in range(n):
        intx += w[i] * dx / 2 * expr.subs(N.x, (1 + x[i]) * dx / 2)
    inty = 0
    for i in range(n):
        inty += w[i] * dy / 2 * intx.subs(N.y, (1 + x[i]) * dy / 2)
    intz = 0
    for i in range(n):
        intz += w[i] * dz / 2 * inty.subs(N.z, (1 + x[i]) * dz / 2)
    return intz

def compile_functional(expr, n_gauss = 3):
    iexpr = integrate(expr, n_gauss)

    # find all named symbols (fields)
    symbs = [
        symb for symb in iexpr.free_symbols if re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
    ]

    # try to reduce multiplications of fields for better performance
    rhs = str(sp.collect(sp.factor_terms(sp.expand(iexpr)), symbs))

    # retrieve topological dimension from first symbol
    match = re.match(r"^_(.*:.*:.*:.*)_$", symbs[0].name)
    shape, idx = [eval(x) for x in match[1].split(':')[2:]]
    dim = len(idx) - len(shape)

    variables = {'dx'}
    for symb in symbs:
        match = re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
        name, space = match[1].split(":")[:2]
        shape, idx = [eval(x) for x in match[1].split(":")[2:]]

        variables.add(name)
        if space == 'node':
            sidx = ','.join(([[':-1','1:'][j] if i < dim else str(j) for i, j in enumerate(idx)]))
            rhs = rhs.replace(symb.name, f"{name}[{sidx}]")
        else:
            if shape == ():
                rhs = rhs.replace(symb.name, f"{name}")
            elif shape == (3,):
                rhs = rhs.replace(symb.name, f"{name}[...,{idx[-1]}]")

    rhs = re.sub(r"_(dx\[\d\])_", r"\1", rhs)
    return rhs, variables

def linear_form_cmds(expr, n_gauss = 3):
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
        vexpr = expr.xreplace(dict([(s, 1.) if s == vsymb else (s, 0.) for s in v]))
        rhs, vvars = compile_functional(vexpr, n_gauss)
        variables = variables.union(vvars)
        vspace, vshape, vidx = v[vsymb]
        if vspace == 'node':
            sidx = ','.join(([[':-1','1:'][j] if i < dim else str(j) for i, j in enumerate(vidx)]))
        else:
            sidx = ','.join(([':' if i < dim else str(j) for i, j in enumerate(vidx)]))
        cmds.append((sidx, rhs))

    return cmds, variables


def gateaux_derivative(expr, var):
    result = []
    for symb in var.free_symbols:
        if not hasattr(symb, "name") or not re.match(r"^_(.*:.*:.*:.*)_$", symb.name):
            continue
        v = sp.Symbol(re.sub(r"^_.*:(.*:.*:.*_)$", r"_v:\1", symb.name))
        result.append(v * expr.diff(symb))
    return reduce(lambda x, y: x+y, result)
