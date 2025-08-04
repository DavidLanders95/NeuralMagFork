# SPDX-License-Identifier: MIT

import inspect
import sys

from scipy import constants

from neuralmag.common import CodeClass, Function, VectorFunction, config
from neuralmag.common import engine as en
from neuralmag.common import logging

__all__ = ["FieldTerm"]


class FieldTerm(CodeClass):
    r"""
    Base class of all effective field contributions. In simple cases,
    a subclass is just required to implement the energy functional of a field
    contribution and a default_name which is used when registering the field
    term with a state. The form compiler of NeuralMag is then used to generate
    efficient code for the computation of both the energy and effective field
    by means of a just-in-time compiler.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Example:
        .. code-block::

            import neuralmag as nm
            from neuralmag.generators import pytorch_generator as gen

            # Example subclass implementing a uniaxial anisotropy
            class UniaxialAnisotropyField(nm.FieldTerm):
                default_name = "uaniso"

                @staticmethod
                def e_expr(m, dim):
                    K = en.Variable("material__Ku", "c" * dim)
                    axis = en.Variable("material__Ku_axis", "c" * dim, (3,))
                    return -K * m.dot(axis) ** 2 * en.dV(dim)

            # Use instance of class to register dynamic attributes in state
            state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
            UniaxialAnisotropyField().register(state)

            # compute field and energy
            h = state.h_uaniso
            E = state.E_uaniso
    """

    default_name = None

    def __init__(self, n_gauss=None):
        self._n_gauss = n_gauss or config.fem["n_gauss"]

    def __init_subclass__(cls, **kwargs):
        if getattr(cls, "default_name") is None:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} without 'default_name' attribute defined"
            )
        return super().__init_subclass__(**kwargs)

    def register(self, state, name=None):
        r"""
        Registers dynamic attributes for the computation of the effective field
        and energy with the given :class:`State` object. By naming convention,
        these methods are registered as :code:`state.h_{name}` and
        :code:`state.E_{name}` or :code:`state.h` and :code:`state.E` in case
        that of :code:`name` being an empty string.

        :param state: The state
        :type state: :class:`State`
        :param name: The name used for the registration, falls back to :code:`default_name`
                     attribute of the class.
        :type name: str, optional
        """
        dim = state.mesh.dim
        if hasattr(self, "e_expr"):
            self.save_and_load_code(self._n_gauss, dim)
        if not hasattr(self, "h"):
            self.h = config.backend.compile(self._code.h)
        if not hasattr(self, "E"):
            self.E = config.backend.compile(self._code.E)
        logging.info_green(
            f"[{self.__class__.__name__}] Register state methods (field:"
            f" '{self.attr_name('h', name)}', energy: '{self.attr_name('E', name)}')"
        )
        setattr(state, self.attr_name("h", name), VectorFunction(state, tensor=self.h))
        setattr(state, self.attr_name("E", name), self.E)

    @classmethod
    def attr_name(cls, attr, name=None):
        r"""
        Returns the attribute name for a given attribute, using the :class:`default_name` attribute
        of the class.

        :param attr: The attribute name, e.g. "E" or "h"
        :type attr: str
        :param name: The name of the class, defaults to :class:`cls.default_name`
        :type attr: str
        """
        if name is None:
            name = cls.default_name
        if name == "":
            return attr
        return f"{attr}_{name}"

    @classmethod
    def _generate_code(cls, n_gauss, dim):
        code = config.backend.CodeBlock()
        m = en.Variable("m", "n" * dim, (3,))

        if not hasattr(cls, "h"):
            # generate linear-form cmds
            if hasattr(cls, "dedm_expr"):
                field_expr = cls.dedm_expr(m, dim)
            else:
                field_expr = en.gateaux_derivative(cls.e_expr(m, dim), m)

            cmds1, vars1 = en.linear_form_cmds(field_expr, n_gauss)

            # generate lumped mass cmds
            v = en.Variable("v", "n" * dim)
            Ms = en.Variable("material__Ms", "c" * dim)
            cmds2, vars2 = en.linear_form_cmds(-constants.mu_0 * Ms * v * en.dV(dim))

            with code.add_function("h", sorted(list(vars1 | vars2 | {"m"}))) as f:
                f.zeros_like("h", "m")
                for cmd in cmds1:
                    f.add_to("h", cmd[0], cmd[1])

                f.zeros_like("mass", "m", shape="m.shape[:-1]")
                for cmd in cmds2:
                    f.add_to("mass", cmd[0], cmd[1])

                f.retrn("h / mass.reshape(mass.shape + (1,))")  # unsqueeze(-1)")

        if not hasattr(cls, "E"):
            terms, variables = en.compile_functional(cls.e_expr(m, dim), n_gauss)
            with code.add_function("E", variables) as f:
                f.retrn_sum(*[term["cmd"] for term in terms])

        return code
