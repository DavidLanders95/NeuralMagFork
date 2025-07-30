# SPDX-License-Identifier: MIT

import jax

from neuralmag.common import config


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


def compile(func):
    if config.jax["jit"]:
        return jax.jit(func)
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

    def assign(self, lhs, rhs, index=None):
        if index is None:
            self.add_line(f"{lhs} = {rhs}")
        else:
            self.add_line(f"{lhs} = {lhs}.at[{index}].set({rhs})")

    def assign_sum(self, lhs, *terms, index=None):
        self.assign(lhs, self.sum(*terms), index)

    def zeros_like(self, var, src, shape=None):
        if shape is None:
            self.add_line(f"{var} = jnp.zeros_like({src})")
        else:
            self.add_line(f"{var} = jnp.zeros({shape}, dtype = {src}.dtype)")

    def add_to(self, var, idx, rhs):
        self.add_line(f"{var} = {var}.at[{idx}].add({rhs})")

    def retrn(self, code):
        self.add_line(f"return {code}")

    def retrn_sum(self, *terms):
        self.add_line(f"return {self.sum(*terms)}")

    def retrn_expanded(self, code, shape):
        self.add_line(f"return jnp.broadcast_to({code}, {shape})")

    def retrn_maximum(self, a, b):
        self.add_line(f"return jnp.maximum({a}, {b})")


class CodeBlock(object):
    def __init__(self, plain=False):
        if plain:
            self._code = ""
        else:
            self._code = "import jax.numpy as jnp\n\n"

    def add_function(self, name, variables):
        return CodeFunction(self, name, variables)

    def add(self, code):
        self._code += code

    def __str__(self):
        return self._code
