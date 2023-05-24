import sympy as sp
import numpy as np
import sympy.vector as sv
import re

import itertools
from itertools import product
from functools import reduce

dx, dy, dz = sp.symbols('_dx[0]_ _dx[1]_ _dx[2]_', real=True, positive=True)
N = sv.CoordSys3D('N')

def Variable(name, space, shape = ()):
    result = []
    if space == 'cg':
        for i,j,k in product([0,1],[0,1],[0,1]):
            phi = (N.x + i * (dx - 2*N.x)) / dx * \
                  (N.y + j * (dy - 2*N.y)) / dy * \
                  (N.z + k * (dz - 2*N.z)) / dz
            if shape == ():
                result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k]}_", real=True) * phi)
            elif shape == (3,):
                for l in range(3):
                    result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k,l]}_", real=True) * phi * [N.i, N.j, N.k][l])
    elif space == 'dg':
        if shape == ():
            result.append(sp.Symbol(f"_{name}:{space}:{shape}:[0,0,0]_", real=True))
        elif shape == (3,):
            for l in range(3):
                result.append(sp.Symbol(f"_{name}:{space}:{shape}:{[0,0,0,l]}_", real=True) * [N.i, N.j, N.k][l])
    else:
        raise NotImplemented
    return reduce(lambda x,y: x+y, result)

def compile_functional(expr):
    variables = set()
    iexpr = sp.integrate(sp.integrate(sp.integrate(expr, (N.x, 0, dx)), (N.y, 0, dy)), (N.z, 0, dz))

    # factor by all coefficients
    csymbs = []
    for symb in iexpr.free_symbols:
        match = re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
        if not match:
            continue
        csymbs.append(symb)
    rhs = str(sp.collect(sp.factor_terms(sp.expand(iexpr)), csymbs))

    for symb in iexpr.free_symbols:
        match = re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
        if not match:
            continue

        name, space = match[1].split(':')[:2]
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
    code = "@jax.jit\n"
    code += f"def assemble_functional(dx, {', '.join(variables)}):\n"
    code += f"    return ({rhs}).sum()\n"
    return code


def linear_form_code(expr):
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
    for vsymb in v:
        vexpr = expr.subs([(s, 1.) if s == vsymb else (s, 0.) for s in v])
        rhs, vvars = compile_functional(vexpr)
        variables = variables.union(vvars)
        vspace, vshape, vidx = v[vsymb]
        if vspace == 'cg':
            sidx = ','.join(([[':-1','1:'][j] if i < 3 else str(j) for i, j in enumerate(vidx)]))
        else:
            sidx = ','.join(([':' if i < 3 else str(j) for i, j in enumerate(vidx)]))
        cmds[sidx] = rhs

    # Assemble Function
    # TODO use @partial(jax.jit, static_argnames=['dx', 'dy', 'dz']) instead? (check speedup)
    code = "@jax.jit\n"
    code += f"def assemble_linear_form(dx, {', '.join(variables)}):\n"
    code += "    result = jnp.zeros_like(m)\n" # TODO m should not be hardcoded
    for sidx, rhs in cmds.items():
        code += f"    result = result.at[{sidx}].add({rhs}, indices_are_sorted=True, unique_indices=True)\n"
    code += "    return result\n"

    return code

def gateaux_derivative(expr, var):
    result = []
    for symb in var.free_symbols:
        if not hasattr(symb, "name") or not re.match(r"^_(.*:.*:.*:.*)_$", symb.name):
            continue
        v = sp.Symbol(re.sub(r"^_.*:(.*:.*:.*_)$", r"_v:\1", symb.name))
        result.append(v * expr.diff(symb))
    return reduce(lambda x,y: x+y, result)

# Exchange
m = Variable('m', 'cg', (3,))
A = Variable('A', 'dg')
energy_expr = A * (
        m.diff(N.x).dot(m.diff(N.x)) + 
        m.diff(N.y).dot(m.diff(N.y)) +
        m.diff(N.z).dot(m.diff(N.z))
        )

print("import jax")
print("import jax.numpy as jnp")
print("")
print(functional_code(energy_expr))
print("")
field_expr = gateaux_derivative(energy_expr, m)
print(linear_form_code(field_expr))

