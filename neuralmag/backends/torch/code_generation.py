# SPDX-License-Identifier: MIT

import torch
from neuralmag.common import config
from neuralmag.common.code_generation import CodeBlockBase, CodeFunctionBase


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
    if config.torch["compile"]:
        return torch.compile(func)
    else:
        return func


class CodeFunction(CodeFunctionBase):
    def zeros(self, name, spaces, shape=()):
        shape_str = self._shape_expr(spaces, shape)
        donor = self._donor
        self.add_line(
            f"{name} = torch.zeros({shape_str}, dtype={donor}.dtype, device="
            f"{donor}.device)"
        )
        self._registry[name] = (spaces, shape)

    def zeros_like(self, var, src, shape=None):
        if shape is None:
            self.add_line(f"{var} = torch.zeros_like({src})")
        else:
            self.add_line(
                f"{var} = torch.zeros({shape}, dtype = {src}.dtype, device ="
                f" {src}.device)"
            )

    def assign(self, lhs, rhs, index=None):
        if index is None:
            self.add_line(f"{lhs} = {rhs}")
        else:
            self.add_line(f"{lhs}[{index}] = {rhs}")

    def add_to(self, var, idx, rhs):
        self.add_line(f"{var}[{idx}] += {rhs}")

    def retrn_expanded(self, code, shape):
        self.add_line(f"return {code}.expand({shape})")

    def retrn_maximum(self, a, b):
        self.add_line(f"return torch.maximum({a}, {b})")


class CodeBlock(CodeBlockBase):
    _code_function_class = CodeFunction

    def __init__(self, plain=False):
        super().__init__("import torch\n\n" if not plain else "")
