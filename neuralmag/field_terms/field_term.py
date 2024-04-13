import inspect
import sys

from scipy import constants

from ..common import Function, VectorFunction, config, logging
from ..generators import pytorch_generator as gen

__all__ = ["FieldTerm"]


class FieldTerm(gen.CodeClass):
    def __init__(self, n_gauss=None):
        self._n_gauss = n_gauss or config.fem["n_gauss"]

    def register(self, state, name=None):
        dim = state.mesh.dim
        if hasattr(self, "e_expr"):
            self.save_and_load_code(self._n_gauss, dim)
        if not hasattr(self, "h"):
            self.h = gen.compile(self._code.h)
        if not hasattr(self, "E"):
            self.E = gen.compile(self._code.E)
        logging.info_green(
            f"[{self.__class__.__name__}] Register state methods (field:"
            f" '{self.attr_name('h', name)}', energy: '{self.attr_name('E', name)}')"
        )
        setattr(state, self.attr_name("h", name), (self.h, "n" * dim, (3,)))
        setattr(state, self.attr_name("E", name), self.E)

    @classmethod
    def attr_name(cls, attr, name=None):
        name = name or cls._name
        if name == "":
            return attr
        return f"{attr}_{name}"

    @classmethod
    def generate_code(cls, n_gauss, dim):
        code = gen.CodeBlock()
        m = gen.Variable("m", "n" * dim, (3,))
        rho = gen.Variable("rho", "c" * dim)

        if not hasattr(cls, "h"):
            # generate linear-form cmds
            # XXX XXX XXX XXX XXX rho!!!!!
            # field_expr = rho * gen.gateaux_derivative(cls.e_expr(m, dim), m)
            field_expr = gen.gateaux_derivative(cls.e_expr(m, dim), m)
            cmds1, vars1 = gen.linear_form_cmds(field_expr, n_gauss)

            # generate lumped mass cmds
            v = gen.Variable("v", "n" * dim)
            Ms = gen.Variable("material__Ms", "c" * dim)
            cmds2, vars2 = gen.linear_form_cmds(
                -constants.mu_0 * rho * Ms * v * gen.dV()
            )

            with code.add_function("h", sorted(list(vars1 | vars2 | {"m"}))) as f:
                f.zeros_like("h", "m")
                for cmd in cmds1:
                    f.add_to("h", cmd[0], cmd[1])

                f.zeros_like("mass", "m", shape="m.shape[:-1]")
                for cmd in cmds2:
                    f.add_to("mass", cmd[0], cmd[1])

                f.retrn("h / mass.unsqueeze(-1)")  # TODO more abstraction?

        if not hasattr(cls, "E"):
            # XXX XXX XXX XXX XXX rho!!!!!
            # terms, variables = gen.compile_functional(rho * cls.e_expr(m, dim), n_gauss)
            terms, variables = gen.compile_functional(cls.e_expr(m, dim), n_gauss)
            rhs = terms[0]["cmd"]  # XXX XXX XXX XXX
            with code.add_function("E", variables) as f:
                f.retrn_sum(rhs)

        return code
