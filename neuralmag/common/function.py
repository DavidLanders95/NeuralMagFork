# SPDX-License-Identifier: MIT

from neuralmag.common import config
from neuralmag.common import engine as en
from neuralmag.common.code_class import CodeClass
from neuralmag.common.code_generation import compile_projection

__all__ = ["Function", "VectorFunction", "CellFunction", "VectorCellFunction"]


class Function(CodeClass):
    """
    This class represents a discretized field on the mesh of a state object.

    If the instance is not intialized with a tensor, the tensor is lazy-
    initialized with zeros on the first access.

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :param spaces: The function spaces in the principal direction (defaults to
        "nnn" for 3D meshes and "nn" for 2D meshes)
    :type spaces: str
    :param shape: The shape of the function (either ``()`` for scalars or ``(3,)`` for
        vectors is currently supported)
    :type shape: tuple
    :param tensor: Tensor with discretized function values or function that depends on state attributes
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    :param dtype: dtype to be used for the tensor
    :type dtype: dtype
    """

    def __init__(
        self, state, spaces=None, shape=(), tensor=None, name=None, dtype=None
    ):
        self._state = state
        self._tensor = None
        self.tensor = tensor

        if spaces is None:
            spaces = "n" * state.mesh.dim
        self._spaces = spaces
        self._shape = shape
        if name is None:
            self._name = "f"
        else:
            self._name = name
        self._dtype = dtype

        tensor_shape = []
        for i, space in enumerate(spaces):
            if space == "c":
                tensor_shape.append(state.mesh.n[i])
            elif space == "n":
                if state.mesh.pbc[i]:
                    tensor_shape.append(state.mesh.n[i])
                else:
                    tensor_shape.append(state.mesh.n[i] + 1)
            else:
                raise Exception(f"Function space '{space}' not supported")

        self._tensor_shape = tuple(tensor_shape) + shape

        self.save_and_load_code(spaces, shape, state.mesh.pbc)

        self._avg = None
        self._avg_on_domain = None
        self._to_cell_fn = None
        self._to_cell_w_fn = None
        self._to_node_fn = None
        self._to_node_w_fn = None

    @property
    def name(self):
        """
        The name of the function
        """
        return self._name

    @property
    def shape(self):
        """
        The shape of the function
        """
        return self._shape

    @property
    def tensor_shape(self):
        """
        The shape of the tensor with the discretized field values
        """
        return self._tensor_shape

    @property
    def state(self):
        """
        The state object used for the construction of the function
        """
        return self._state

    @property
    def spaces(self):
        """
        The function spaces of the function
        """
        return self._spaces

    @property
    def func(self):
        if callable(self._tensor):
            return self._tensor
        return None

    @property
    def func_id(self):
        return f"function__{str(id(self))}"

    @property
    def tensor(self):
        """
        The tensor containing the discretized values of the function
        """
        if callable(self._tensor):
            return getattr(self._state, self.func_id)

        if self._tensor is None:
            dtype = self._dtype or self._state.dtype
            self._tensor = config.backend.zeros(
                self._tensor_shape, dtype=dtype, device=self._state.device
            )
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        if self.func:
            delattr(self._state, self.func_id)

        self._tensor = tensor

        if self.func:
            setattr(self._state, self.func_id, self.func)

    def fill(self, constant, expand=False):
        """
        Fills the tensor of the function with a constant value.

        :param constant: The constant to fill the tensor
        :type constant: int, list
        :param expand: If True, the tensor is set by expanding the constant
            to the size of the mesh using :code:`torch.Tensor.expand`
            resulting in minimal storage consumption.
        :type expand: bool
        :return: The function itself
        :rvalue: :class:`Function`

        :Example:
            .. code-block::

                state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
                f = nm.Function(state, shape = (3,)).fill([1.0, 2.0, 3.0])
        """
        tensor = self.state.tensor(constant, dtype=self._dtype)
        shape = self._tensor_shape
        if self.shape == (3,) and expand == False:
            shape = self._tensor_shape[:-1] + (1,)

        if expand:
            self.tensor = config.backend.broadcast_to(tensor, shape)
        else:
            self.tensor = config.backend.tile(tensor, shape)
        return self

    def fill_by_domain(self, values):
        aux = self.state.tensor(values)
        if self._spaces == "c" * self._state.mesh.dim:
            self.tensor = lambda domains: aux[domains]
        else:
            raise NotImplementedError

        return self

    def avg(self, domain_id=None):
        """
        Returns the componentwise average of the function over the mesh.

        :param domain_id: id of domain to average over, if None average is computed over all domains with id > 0
        :type domain_id: int
        :return: The componentwise average
        :rtype: :class:`torch.Tensor`
        """
        if domain_id is None:
            if self._avg is None:
                self._avg = self._state.resolve(self._code.avg, ["f", "domains"])
            return self._avg(self.tensor, self._state.domains.tensor)
        else:
            if self._avg_on_domain is None:
                self._avg_on_domain = self._state.resolve(
                    self._code.avg,
                    ["f", "domains", "domain_id"],
                    remap={"domain": "subdomain"},
                )
            return self._avg_on_domain(
                self.tensor, self._state.domains.tensor, domain_id
            )

    def _make_function(self, spaces, tensor):
        """Create a new Function with the given spaces and concrete tensor."""
        if spaces == "c" * len(spaces) and self._shape == (3,):
            return VectorCellFunction(self._state, tensor=tensor)
        elif spaces == "c" * len(spaces):
            return CellFunction(self._state, tensor=tensor, shape=self._shape)
        elif self._shape == (3,):
            return VectorFunction(self._state, tensor=tensor)
        else:
            return Function(
                self._state, spaces=spaces, shape=self._shape, tensor=tensor
            )

    def to_cell(self, weight=None):
        """
        Project this function to cell space via mass-lumped L² projection.

        :param weight: Optional scalar cell-based weight for a weighted projection.
        :type weight: :class:`Function`, optional
        :return: The projected function in cell space
        :rtype: :class:`Function`
        """
        dim = len(self._spaces)
        if set(self._spaces) == {"c"}:
            return self
        if weight is not None:
            if self._to_cell_w_fn is None:
                self._to_cell_w_fn = self._state.resolve(
                    self._code.to_cell_w, ["f", "weight"]
                )
            return self._make_function(
                "c" * dim, self._to_cell_w_fn(self.tensor, weight.tensor)
            )
        if self._to_cell_fn is None:
            self._to_cell_fn = self._state.resolve(self._code.to_cell, ["f"])
        return self._make_function("c" * dim, self._to_cell_fn(self.tensor))

    def to_node(self, weight=None):
        """
        Project this function to node space via mass-lumped L² projection.

        :param weight: Optional scalar cell-based weight for a weighted projection.
        :type weight: :class:`Function`, optional
        :return: The projected function in node space
        :rtype: :class:`Function`
        """
        dim = len(self._spaces)
        if set(self._spaces) == {"n"}:
            return self
        if weight is not None:
            if self._to_node_w_fn is None:
                self._to_node_w_fn = self._state.resolve(
                    self._code.to_node_w, ["f", "weight"]
                )
            return self._make_function(
                "n" * dim, self._to_node_w_fn(self.tensor, weight.tensor)
            )
        if self._to_node_fn is None:
            self._to_node_fn = self._state.resolve(self._code.to_node, ["f"])
        return self._make_function("n" * dim, self._to_node_fn(self.tensor))

    @classmethod
    def _generate_code(cls, spaces, shape, pbc=None):
        code = config.backend.CodeBlock(pbc=pbc)
        dim = len(spaces)

        # generate avg method
        f = en.Variable("f", spaces, shape)
        with code.add_function("avg", ["rho", "dx", "f"]) as func:
            terms, _ = en.compile_functional(1 * en.dV(dim), pbc=pbc)
            func.assign_sum("vol", *[term["cmd"] for term in terms])

            if shape == ():
                terms, variables = en.compile_functional(f * en.dV(dim), pbc=pbc)
                func.assign_sum("fint", *[term["cmd"] for term in terms])
            elif shape == (3,):
                func.zeros_like("fint", "f", (3,))
                for i in range(3):
                    terms, _ = en.compile_functional(
                        f.dot(en.cs_e[i]) * en.dV(dim), pbc=pbc
                    )
                    func.assign_sum("fint", *[term["cmd"] for term in terms], index=i)

            func.retrn("fint / vol")

        # generate to_node / to_cell projection methods
        node_spaces = "n" * dim
        cell_spaces = "c" * dim
        for direction, fname in [("c2n", "to_node"), ("n2c", "to_cell")]:
            src_sp = cell_spaces if direction == "c2n" else node_spaces
            tgt_sp = node_spaces if direction == "c2n" else cell_spaces

            for weighted in [False, True]:
                fn_name = fname + ("_w" if weighted else "")
                weight_name = "weight" if weighted else None

                field_cmds, field_vars, mass_cmds, mass_vars = compile_projection(
                    direction,
                    dim,
                    shape,
                    "f",
                    pbc=pbc,
                    weight_name=weight_name,
                )
                all_vars = sorted(field_vars | mass_vars)
                var_spaces = {"f": (src_sp, shape)}
                if weighted:
                    var_spaces["weight"] = cell_spaces

                with code.add_function(
                    fn_name, all_vars, var_spaces=var_spaces
                ) as func:
                    func.zeros("result", tgt_sp, shape)
                    for idx, rhs in field_cmds:
                        func.add_to("result", idx, rhs)
                    func.zeros("mass", tgt_sp)
                    for idx, rhs in mass_cmds:
                        func.add_to("mass", idx, rhs)
                    if shape == (3,):
                        func.retrn("result / mass.reshape(mass.shape + (1,))")
                    else:
                        func.retrn("result / mass")

        return code


class CellFunction(Function):
    """
    Subclass of :class:`Function` with the function space set to cellwise in each dimension.

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :type spaces: str
    :param shape: The shape of the function (either ``()`` for scalars or ``(3,)`` for
        vectors is currently supported)
    :type shape: tuple
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, **kwargs):
        assert "spaces" not in kwargs
        kwargs["spaces"] = "c" * state.mesh.dim
        super().__init__(state, **kwargs)


class VectorFunction(Function):
    """
    Subclass of :class:`Function` with the shape set to (3,).

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :param spaces: The function spaces in the principal direction (defaults to
        "nnn" for 3D meshes and "nn" for 2D meshes)
    :type spaces: str
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, **kwargs):
        assert "shape" not in kwargs
        kwargs["shape"] = (3,)
        super().__init__(state, **kwargs)


class VectorCellFunction(Function):
    """
    Subclass of :class:`Function` with the shape set to (3,) and the
    function space set to cellwise in each dimension.

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, **kwargs):
        assert "spaces" not in kwargs
        assert "shape" not in kwargs
        kwargs["spaces"] = "c" * state.mesh.dim
        kwargs["shape"] = (3,)
        super().__init__(state, **kwargs)
