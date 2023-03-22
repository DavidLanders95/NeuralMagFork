import sympy as sp
import sympy.vector as sv
import re

from itertools import product

dx, dy, dz = sp.symbols('_dx_[0] _dx_[1] _dx_[2]')
N = sv.CoordSys3D('N')

def Variable(name, space, shape = ()):
    if shape == ():
        expr = 0
    elif shape == (3,):
        expr = sp.vector.Vector.zero
    else:
        raise NotImplemented

    if space == 'cg':
        for i,j,k in product([0,1],[0,1],[0,1]):
            phi = (N.x + i * (dx - 2*N.x)) / dx * \
                  (N.y + j * (dy - 2*N.y)) / dy * \
                  (N.z + k * (dz - 2*N.z)) / dz
            if shape == ():
                expr += sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k]}_") * phi
            elif shape == (3,):
                for l in range(3):
                    expr += sp.Symbol(f"_{name}:{space}:{shape}:{[i,j,k,l]}_") * phi * [N.i, N.j, N.k][l]
    elif space == 'dg':
        if shape == ():
            expr = sp.Symbol(f"_{name}:{space}:{shape}:[0,0,0]_")
        elif shape == (3,):
            for l in range(3):
                expr += sp.Symbol(f"_{name}:{space}:{shape}:{[0,0,0,l]}_") * [N.i, N.j, N.k][l]
    else:
        raise NotImplemented
    return expr


def kernel_cmds(expr):
    cmds = {}
    v = {}

    # collect all test functions in expr
    for symb in sorted(list(expr.free_symbols), key = lambda s: s.name):
        match = re.match(r"^_v:(.*:.*:.*)_$", symb.name)
        if match:
            v[symb] = match[1].split(':')
            v[symb][1:] = [eval(x) for x in v[symb][1:]]

    # process test functions
    for vsymb in v:
        vexpr = expr.subs([(s, 1.) if s == vsymb else (s, 0.) for s in v])
        iexpr = sp.factor(sp.integrate(sp.integrate(sp.integrate(vexpr, (N.x, 0, dx)), (N.y, 0, dy)), (N.z, 0, dz)))
        rhs = str(iexpr)

        vspace, vshape, vidx = v[vsymb]
        for symb in iexpr.free_symbols:
            match = re.match(r"^_(.*:.*:.*:.*)_$", symb.name)
            if not match:
                continue

            name, space = match[1].split(':')[:2]
            shape, idx = [eval(x) for x in match[1].split(':')[2:]]

            if space == 'cg':
                sidx = ','.join(([[':-1','1:'][j] if i < 3 else str(j) for i, j in enumerate(idx)]))
            else:
                sidx = ','.join(([':' if i < 3 else str(j) for i, j in enumerate(idx)]))
            rhs = rhs.replace(symb.name, f"{name}[{sidx}]")

        if vspace == 'cg':
            sidx = ','.join(([[':-1','1:'][j] if i < 3 else str(j) for i, j in enumerate(vidx)]))
        else:
            sidx = ','.join(([':' if i < 3 else str(j) for i, j in enumerate(vidx)]))
        lhs = f"_result_[{sidx}]"

        if lhs not in cmds:
            cmds[lhs] = []
        cmds[lhs].append(rhs)

    return cmds

def kernel_code(expr):
    code = ""
    cmds = kernel_cmds(expr)
    for lhs, rhs in cmds.items():
        code += f"{lhs} += {' + '.join(rhs)}\n"
    #for lhs, rhs in cmds.items():
    #    if condition is None:
    #        for cmd in cmdlist:
    #            code += cmd + "\n"
    #    else:
    #        code += "if " + condition + ":\n"
    #        for cmd in cmdlist:
    #            code += "    " + cmd + "\n"

    return code

# Exchange
v = Variable('v', 'cg', (3,))
m = Variable('m', 'cg', (3,))
A = Variable('A', 'dg')
expr = 2. * A * (m.diff(N.x).dot(v.diff(N.x)) + \
                 m.diff(N.y).dot(v.diff(N.y)) + \
                 m.diff(N.z).dot(v.diff(N.z)))

# Anisotropy
#v = Variable('v', 'cg', (3,))
#m = Variable('m', 'cg', (3,))
#K = Variable('K', 'dg')
#Kaxis = Variable('Kaxis', 'dg', (3,))
#expr = - 2. * K * (Kaxis.dot(m) * Kaxis.dot(v)) 

# MASS Matrix
#v = Variable('v', 'cg', (3,))
#Ms = Variable('Ms', 'dg', (3,))
#expr = Ms.dot(v)

print(kernel_code(expr))
