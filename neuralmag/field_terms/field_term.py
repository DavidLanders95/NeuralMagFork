# SPDX-License-Identifier: MIT


from scipy import constants

from neuralmag.common import CodeClass, Function, config, logging
from neuralmag.common import engine as en

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
                def e_expr(m, dim, _options):
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
        self._options = None

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
        m_spaces = state.m.spaces

        if set(m_spaces) == {"c"}:
            self._options = {**(self._options or {}), "m_spaces": m_spaces}

        if hasattr(self, "e_expr") and self.e_expr is not None:
            self.save_and_load_code(self._n_gauss, dim, self._options, m_spaces)
        if not hasattr(self, "h"):
            self.h = config.backend.compile(self._code.h)
        if not hasattr(self, "e"):
            self.e = config.backend.compile(self._code.e)
        if not hasattr(self, "E"):
            self.E = config.backend.compile(self._code.E)

        logging.info_green(
            f"[{self.__class__.__name__}] Register state methods (field:"
            f" '{self.attr_name('h', name)}', energy: '{self.attr_name('E', name)}', energy density: '{self.attr_name('e', name)}')"
        )
        setattr(
            state,
            self.attr_name("h", name),
            Function(state, spaces=m_spaces, shape=(3,), tensor=self.h),
        )
        setattr(
            state,
            self.attr_name("e", name),
            Function(state, spaces=m_spaces, tensor=self.e),
        )
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
    def _generate_code(cls, n_gauss, dim, options, m_spaces="n"):
        is_cell = set(m_spaces) == {"c"}

        code = config.backend.CodeBlock()
        m = en.Variable("m", "n" * dim, (3,))

        # Compile cell↔node projection forms for FIC
        if is_cell:
            v_n = en.Variable("v", "n" * dim, (3,))
            m_c = en.Variable("m", "c" * dim, (3,))
            c2n_cmds, c2n_vars = en.linear_form_cmds(v_n.dot(m_c) * en.dV(dim), 1)
            v_n_s = en.Variable("v", "n" * dim)
            c2n_mass_cmds, c2n_mass_vars = en.linear_form_cmds(v_n_s * en.dV(dim), 1)

            v_c = en.Variable("v", "c" * dim, (3,))
            h_n = en.Variable("h_fem", "n" * dim, (3,))
            n2c_cmds, n2c_vars = en.linear_form_cmds(v_c.dot(h_n) * en.dV(dim), 1)
            v_c_s = en.Variable("v", "c" * dim)
            n2c_mass_cmds, n2c_mass_vars = en.linear_form_cmds(v_c_s * en.dV(dim), 1)

            # Scalar n2c for energy density
            e_n = en.Variable("e_fem", "n" * dim)
            n2c_e_cmds, n2c_e_vars = en.linear_form_cmds(v_c_s * e_n * en.dV(dim), 1)

        if not hasattr(cls, "h"):
            if hasattr(cls, "dedm_expr"):
                field_expr = cls.dedm_expr(m, dim, options)
            else:
                field_expr = en.gateaux_derivative(cls.e_expr(m, dim, options), m)

            cmds1, vars1 = en.linear_form_cmds(field_expr, n_gauss)

            v = en.Variable("v", "n" * dim)
            Ms = en.Variable("material__Ms", "c" * dim)
            cmds2, vars2 = en.linear_form_cmds(-constants.mu_0 * Ms * v * en.dV(dim))

            all_vars = vars1 | vars2 | {"m"}
            if is_cell:
                all_vars |= c2n_vars | c2n_mass_vars | n2c_vars | n2c_mass_vars
                all_vars -= {"h_fem", "e_fem"}

            with code.add_function("h", sorted(list(all_vars))) as f:
                if is_cell:
                    # Cell→node projection of m
                    f.zeros_like(
                        "m_node",
                        "m",
                        shape=f"({','.join(f'm.shape[{i}]+1' for i in range(dim))},3)",
                    )
                    for cmd in c2n_cmds:
                        f.add_to("m_node", cmd[0], cmd[1])
                    f.zeros_like(
                        "c2n_mass",
                        "m",
                        shape=f"({','.join(f'm.shape[{i}]+1' for i in range(dim))},)",
                    )
                    for cmd in c2n_mass_cmds:
                        f.add_to("c2n_mass", cmd[0], cmd[1])
                    f.add_line("m = m_node / c2n_mass.reshape(c2n_mass.shape + (1,))")

                # Standard FEM h computation
                f.zeros_like("h", "m")
                for cmd in cmds1:
                    f.add_to("h", cmd[0], cmd[1])

                f.zeros_like("mass", "m", shape="m.shape[:-1]")
                for cmd in cmds2:
                    f.add_to("mass", cmd[0], cmd[1])
                f.add_line("h = h / mass.reshape(mass.shape + (1,))")

                if is_cell:
                    # Node→cell projection of h
                    f.add_line("h_fem = h")
                    f.zeros_like(
                        "h_cell",
                        "m",
                        shape=f"({','.join(f'h_fem.shape[{i}]-1' for i in range(dim))},3)",
                    )
                    for cmd in n2c_cmds:
                        f.add_to("h_cell", cmd[0], cmd[1])
                    f.zeros_like(
                        "n2c_mass",
                        "m",
                        shape=f"({','.join(f'h_fem.shape[{i}]-1' for i in range(dim))},)",
                    )
                    for cmd in n2c_mass_cmds:
                        f.add_to("n2c_mass", cmd[0], cmd[1])
                    f.add_line("h = h_cell / n2c_mass.reshape(n2c_mass.shape + (1,))")

                f.retrn("h")

        if not hasattr(cls, "e"):
            v = en.Variable("v", "n" * dim)

            cmds1, vars1 = en.linear_form_cmds(v * cls.e_expr(m, dim, options), n_gauss)
            cmds2, vars2 = en.linear_form_cmds(v * en.dV(dim))

            all_vars = vars1 | vars2 | {"m"}
            if is_cell:
                all_vars |= c2n_vars | c2n_mass_vars | n2c_e_vars | n2c_mass_vars
                all_vars -= {"h_fem", "e_fem"}

            with code.add_function("e", sorted(list(all_vars))) as f:
                if is_cell:
                    # Cell→node projection of m
                    f.zeros_like(
                        "m_node",
                        "m",
                        shape=f"({','.join(f'm.shape[{i}]+1' for i in range(dim))},3)",
                    )
                    for cmd in c2n_cmds:
                        f.add_to("m_node", cmd[0], cmd[1])
                    f.zeros_like(
                        "c2n_mass",
                        "m",
                        shape=f"({','.join(f'm.shape[{i}]+1' for i in range(dim))},)",
                    )
                    for cmd in c2n_mass_cmds:
                        f.add_to("c2n_mass", cmd[0], cmd[1])
                    f.add_line("m = m_node / c2n_mass.reshape(c2n_mass.shape + (1,))")

                # Standard FEM e computation
                f.zeros_like("e", "m", shape="m.shape[:-1]")
                for cmd in cmds1:
                    f.add_to("e", cmd[0], cmd[1])

                f.zeros_like("mass", "m", shape="m.shape[:-1]")
                for cmd in cmds2:
                    f.add_to("mass", cmd[0], cmd[1])
                f.add_line("e = e / mass")

                if is_cell:
                    # Node→cell projection of e
                    f.add_line("e_fem = e")
                    f.zeros_like(
                        "e_cell",
                        "m",
                        shape=f"({','.join(f'e_fem.shape[{i}]-1' for i in range(dim))},)",
                    )
                    for cmd in n2c_e_cmds:
                        f.add_to("e_cell", cmd[0], cmd[1])
                    f.zeros_like(
                        "n2c_mass",
                        "m",
                        shape=f"({','.join(f'e_fem.shape[{i}]-1' for i in range(dim))},)",
                    )
                    for cmd in n2c_mass_cmds:
                        f.add_to("n2c_mass", cmd[0], cmd[1])
                    f.add_line("e = e_cell / n2c_mass")

                f.retrn("e")

        if not hasattr(cls, "E"):
            terms, variables = en.compile_functional(
                cls.e_expr(m, dim, options), n_gauss
            )
            all_vars = variables
            if is_cell:
                all_vars = variables | c2n_vars | c2n_mass_vars

            with code.add_function("E", all_vars) as f:
                if is_cell:
                    # Cell→node projection of m
                    f.zeros_like(
                        "m_node",
                        "m",
                        shape=f"({','.join(f'm.shape[{i}]+1' for i in range(dim))},3)",
                    )
                    for cmd in c2n_cmds:
                        f.add_to("m_node", cmd[0], cmd[1])
                    f.zeros_like(
                        "c2n_mass",
                        "m",
                        shape=f"({','.join(f'm.shape[{i}]+1' for i in range(dim))},)",
                    )
                    for cmd in c2n_mass_cmds:
                        f.add_to("c2n_mass", cmd[0], cmd[1])
                    f.add_line("m = m_node / c2n_mass.reshape(c2n_mass.shape + (1,))")

                f.retrn_sum(*[term["cmd"] for term in terms])

        return code
