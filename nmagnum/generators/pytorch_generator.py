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

dx, dy, dz = sp.symbols('_dx[0]_ _dx[1]_ _dx[2]_', real=True, positive=True)
N = sv.CoordSys3D('N')

class CodeClass(object):
    def __init__(self, *args, generate_code = True):
        if not generate_code:
            return

        this_module = pathlib.Path(importlib.import_module(self.__module__).__file__)
        code_file_path = this_module.parent / 'code' / f"{this_module.stem}_code.py"

        # generate code
        if not code_file_path.is_file():
            code_file_path.parent.mkdir(parents = True, exist_ok = True)
            # TODO check if generate_code method exists
            code = self.generate_code()
            with open(code_file_path, 'w') as f:
                f.write(code)

        # import code
        module_spec = importlib.util.spec_from_file_location('code', code_file_path)
        self.code = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.code)

def Variable(name, space, shape = ()):
    result = []
    if space == 'cg':
        for i,j,k in product([0,1],[0,1],[0,1]):
            phi = (1 - N.x/dx + 2*i*N.x/dx - i) * \
                  (1 - N.y/dy + 2*j*N.y/dy - j) * \
                  (1 - N.z/dz + 2*k*N.z/dz - k)

            if shape == ():
                result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k]}_", real=True) * phi)
            elif shape == (3,):
                for l in range(3):
                    result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k,l]}_", real=True) * phi * [N.i, N.j, N.k][l])
    elif space == 'dg':
        if shape == ():
            result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[0,0,0]}_", real=True))
        elif shape == (3,):
            for l in range(3):
                result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[0,0,0,l]}_", real=True) * [N.i, N.j, N.k][l])
    else:
        raise NotImplemented
    return reduce(lambda x,y: x+y, result)


def integrate(expr, n = 3):
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

def compile_functional(expr):
    iexpr = integrate(expr)

    # find all named symbols (fields)
    symbs = [symb for symb in iexpr.free_symbols if re.match(r"^_(.*:.*:.*:.*)_$", symb.name)]

    # try to reduce multiplications of fields for better performance
    rhs = str(sp.collect(sp.factor_terms(sp.expand(iexpr)), symbs))

    variables = set()
    for symb in symbs:
        match = re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
        name = match[1].split(':')[0].replace('.', '__')
        space = match[1].split(':')[1]
        #name, space = match[1].split(':')[:2]
        shape, idx = [eval(x) for x in match[1].split(':')[2:]]

        variables.add(name)
        if space == 'cg':
            sidx = ','.join(([[':-1','1:'][j] if i < 3 else str(j) for i, j in enumerate(idx)]))
        else:
            sidx = ','.join(([':' if i < 3 else str(j) for i, j in enumerate(idx)]))
        rhs = rhs.replace(symb.name, f"{name}[{sidx}]")

    rhs = re.sub(r"_(dx\[\d\])_", r"\1", rhs)
    return rhs, variables

def functional_code(expr):
    rhs, variables = compile_functional(expr)

    # Assemble Function
    code = "@torch.compile\n"
    code += f"def assemble_functional(dx, {', '.join(variables)}):\n"
    code += f"    return ({rhs}).sum()\n"
    return code


def linear_form_cmds(expr, result_name = 'result'):
    cmds = {}
    v = {}

    # collect all test functions in expr
    for symb in sorted(list(expr.free_symbols), key = lambda s: s.name):
        match = re.match(r"^_v:(.*:.*:.*)_$", symb.name)
        if match:
            v[symb] = match[1].split(':')
            v[symb][1:] = [eval(x) for x in v[symb][1:]]

    # process test functions
    variables = set()
    for vsymb in tqdm(v, desc = "Generating..."):
        vexpr = expr.xreplace(dict([(s, 1.) if s == vsymb else (s, 0.) for s in v]))
        rhs, vvars = compile_functional(vexpr)
        variables = variables.union(vvars)
        vspace, vshape, vidx = v[vsymb]
        if vspace == 'cg':
            sidx = ','.join(([[':-1','1:'][j] if i < 3 else str(j) for i, j in enumerate(vidx)]))
        else:
            sidx = ','.join(([':' if i < 3 else str(j) for i, j in enumerate(vidx)]))
        lhs = f"{result_name}[{sidx}]"
        cmds[lhs] = rhs

    return cmds, variables

def linear_form_code(expr):
    cmds, variables = linear_form_cmds(expr, 'result')

    # Assemble Function
    code = "@torch.compile\n"
    code += f"def assemble_linear_form(result, dx, {', '.join(variables)}):\n"
    code += "    result[:] = 0\n"
    for lhs, rhs in cmds.items():
        code += f"    {lhs} += {rhs}\n"

    return code

def gateaux_derivative(expr, var):
    result = []
    for symb in var.free_symbols:
        if not hasattr(symb, "name") or not re.match(r"^_(.*:.*:.*:.*)_$", symb.name):
            continue
        v = sp.Symbol(re.sub(r"^_.*:(.*:.*:.*_)$", r"_v:\1", symb.name))
        result.append(v * expr.diff(symb))
    return reduce(lambda x,y: x+y, result)
