# SPDX-License-Identifier: MIT

from neuralmag.common import config
from neuralmag.common import engine as en
from neuralmag.common.code_class import CodeClass

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
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, spaces=None, shape=(), tensor=None, name=None):
        self._state = state
        if spaces is None:
            spaces = "n" * state.mesh.dim
        self._spaces = spaces
        self._shape = shape
        if name is None:
            self._name = "f"
        else:
            self._name = name

        tensor_shape = []
        for i, space in enumerate(spaces):
            if space == "c":
                tensor_shape.append(state.mesh.n[i])
            elif space == "n":
                tensor_shape.append(state.mesh.n[i] + 1)
            else:
                raise Exception(f"Function space '{space}' not supported")

        self._tensor_shape = tuple(tensor_shape) + shape

        if tensor is None or isinstance(tensor, config.backend.Tensor):
            self._tensor = tensor
        else:
            raise NotImplemented("Unsupported tensor type.")
        self._expanded = False

        self.save_and_load_code(spaces, shape)

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
    def tensor(self):
        """
        The tensor containing the discretized values of the function
        """
        if self._tensor is None:
            self._tensor = config.backend.zeros(
                self._tensor_shape, dtype=self._state.dtype, device=self._state.device
            )
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor

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
        tensor = self.state.tensor(constant)
        shape = self._tensor_shape
        if self.shape == (3,) and expand == False:
            shape = self._tensor_shape[:-1] + (1,)

        if expand:
            self._tensor = config.backend.broadcast_to(tensor, shape)
        else:
            self._tensor = config.backend.tile(tensor, shape)
        return self

    def avg(self):
        """
        Returns the componentwise average of the function over the mesh.

        :return: The componentwise average
        :rtype: :class:`torch.Tensor`
        """
        return self._code.avg(self._state.rho.tensor, self._state.dx, self.tensor)

    @classmethod
    def _generate_code(cls, spaces, shape):
        code = config.backend.CodeBlock()
        dim = len(spaces)

        # generate avg method
        f = en.Variable("f", spaces, shape)
        with code.add_function("avg", ["rho", "dx", "f"]) as func:
            terms, _ = en.compile_functional(1 * en.dV(dim))
            func.assign_sum("vol", *[term["cmd"] for term in terms])

            if shape == ():
                terms, variables = en.compile_functional(f * en.dV(dim))
                func.assign_sum("fint", *[term["cmd"] for term in terms])
            elif shape == (3,):
                func.zeros_like("fint", "f", (3,))
                for i in range(3):
                    terms, _ = en.compile_functional(f.dot(en.cs_e[i]) * en.dV(dim))
                    func.assign_sum(f"fint", *[term["cmd"] for term in terms], index=i)

            func.retrn("fint / vol")

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
