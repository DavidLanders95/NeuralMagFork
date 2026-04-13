# SPDX-License-Identifier: MIT

from neuralmag.common import engine as en

# Cache of compiled mass-lumped projection forms used by
# ``CodeFunctionBase.to_node`` / ``CodeFunctionBase.to_cell``. Keyed by
# ``(direction, dim, trailing_shape, src_name)``.
_projection_cache = {}


def compile_projection(direction, dim, trailing_shape, src_name, pbc, weight_name=None):
    """
    Compile the mass-lumped projection between cell- and node-based
    representations.

    :param direction: ``"c2n"`` or ``"n2c"``
    :param dim: mesh topological dimension
    :param trailing_shape: ``()`` for scalar fields or ``(3,)`` for vectors
    :param src_name: name of the source variable embedded in the returned cmds
    :param pbc: periodic boundary condition tuple
    :param weight_name: optional name of a scalar cell-based weight variable.
        When set, the projection computes the weighted L² projection
        ``∫ w·v·f dV / ∫ w·v dV`` instead of ``∫ v·f dV / ∫ v dV``.
    :return: ``(field_cmds, field_vars, mass_cmds, mass_vars)``
    """
    key = (direction, dim, trailing_shape, src_name, pbc, weight_name)
    if key in _projection_cache:
        return _projection_cache[key]

    src_spaces = "c" * dim if direction == "c2n" else "n" * dim
    tgt_spaces = "n" * dim if direction == "c2n" else "c" * dim

    w = en.Variable(weight_name, "c" * dim) if weight_name else 1

    v = en.Variable("v", tgt_spaces, trailing_shape)
    src = en.Variable(src_name, src_spaces, trailing_shape)
    if trailing_shape == (3,):
        form = w * v.dot(src) * en.dV(dim)
    else:
        form = w * v * src * en.dV(dim)
    field_cmds, field_vars = en.linear_form_cmds(form, 1, pbc=pbc)

    v_s = en.Variable("v", tgt_spaces)
    mass_cmds, mass_vars = en.linear_form_cmds(w * v_s * en.dV(dim), 1, pbc=pbc)

    _projection_cache[key] = (field_cmds, field_vars, mass_cmds, mass_vars)
    return _projection_cache[key]


class CodeFunctionBase(object):
    """
    Base class for backend-specific code-function builders. Provides
    space-aware tensor allocation, cell/node projection primitives, and
    common emit helpers. Subclasses must override the backend-specific
    methods: ``zeros``, ``zeros_like``, ``assign``, ``add_to``,
    ``retrn_expanded``, ``retrn_maximum``.
    """

    def __init__(self, block, name, variables, var_spaces=None):
        self._block = block
        self._name = name
        self._pbc = block._pbc
        # argument list: deduped, insertion order preserved
        self._variables = list(dict.fromkeys(variables))
        # registry of arg-name -> (spaces, trailing_shape); populated when the
        # caller passes ``var_spaces``. Enables space-aware allocation
        # (``zeros``) and cell<->node projection (``to_node`` / ``to_cell``).
        self._registry = {}
        self._donor = None
        if var_spaces:
            for k, v in var_spaces.items():
                if isinstance(v, str):
                    spaces, shape = v, ()
                else:
                    spaces, shape = v
                self._registry[k] = (spaces, shape)
                if self._donor is None:
                    self._donor = k

    def __enter__(self):
        # Buffer the body; the ``def`` signature is emitted on ``__exit__``
        # so that ``to_node`` / ``to_cell`` may lazily extend the argument
        # list if they reference variables the caller did not list upfront.
        self._body = ""
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return False
        signature = f"def {self._name}({', '.join(self._variables)}):\n"
        self._block.add(signature + self._body)
        self._block.add("\n")
        return True

    @staticmethod
    def sum(*terms):
        return " + ".join([f"({term}).sum()" for term in terms])

    def add_line(self, code):
        self._body += f"    {code}\n"

    def _ensure_vars(self, names):
        # Only add names that are neither already declared as arguments nor
        # known as local allocations (tracked in the space registry). This
        # prevents projection cmds from accidentally pulling a locally
        # allocated tensor (e.g. ``h`` during ``to_cell``) into the function
        # signature.
        for n in names:
            if n not in self._variables and n not in self._registry:
                self._variables.append(n)

    def rebind(self, name, new_spaces):
        """
        Update the registered function space of ``name`` after an in-place
        reassignment. Pure metadata update; emits no code.
        """
        _, shape = self._registry[name]
        self._registry[name] = (new_spaces, shape)

    def _shape_expr(self, target_spaces, trailing_shape):
        """
        Build a tuple literal for a zeros call that describes a tensor of
        function space ``target_spaces`` with trailing ``trailing_shape``,
        derived from a registered reference variable.
        """
        dim = len(target_spaces)
        ref_name, (ref_spaces, _) = self._reference_for(dim)
        parts = []
        for i in range(dim):
            # With PBC, node and cell spaces have the same size (n[i]),
            # so the delta between them is 0.
            if self._pbc and self._pbc[i]:
                delta = 0
            else:
                delta = (1 if target_spaces[i] == "n" else 0) - (1 if ref_spaces[i] == "n" else 0)
            if delta == 0:
                parts.append(f"{ref_name}.shape[{i}]")
            elif delta > 0:
                parts.append(f"{ref_name}.shape[{i}]+{delta}")
            else:
                parts.append(f"{ref_name}.shape[{i}]{delta}")
        for s in trailing_shape:
            parts.append(str(s))
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    def _reference_for(self, dim):
        for name, (spaces, shape) in self._registry.items():
            if len(spaces) == dim:
                return name, (spaces, shape)
        raise RuntimeError(
            f"No reference variable with dim={dim} registered on CodeFunction"
            f" '{self._name}'. Did you forget to pass var_spaces?"
        )

    def to_spaces(self, name, target_spaces):
        """
        Project ``name`` into ``target_spaces`` (all ``'n'`` or all ``'c'``).
        Dispatches to ``to_node`` or ``to_cell``.
        """
        if set(target_spaces) == {"n"}:
            self.to_node(name)
        elif set(target_spaces) == {"c"}:
            self.to_cell(name)
        else:
            raise ValueError(f"to_spaces expects homogeneous target spaces, got '{target_spaces}'")

    def to_node(self, name):
        """
        Project ``name`` into the node-based function space via mass-lumped
        projection. No-op if ``name`` is already node-based.
        """
        current_spaces, trailing = self._registry[name]
        dim = len(current_spaces)
        target = "n" * dim
        if current_spaces == target:
            return
        assert current_spaces == "c" * dim, f"to_node expects fully-cell or fully-node spaces, got '{current_spaces}'"
        self._emit_projection(name, "c2n", dim, trailing)
        self.rebind(name, target)

    def to_cell(self, name):
        """
        Project ``name`` into the cell-based function space via mass-lumped
        projection. No-op if ``name`` is already cell-based.
        """
        current_spaces, trailing = self._registry[name]
        dim = len(current_spaces)
        target = "c" * dim
        if current_spaces == target:
            return
        assert current_spaces == "n" * dim, f"to_cell expects fully-cell or fully-node spaces, got '{current_spaces}'"
        self._emit_projection(name, "n2c", dim, trailing)
        self.rebind(name, target)

    def _emit_projection(self, name, direction, dim, trailing):
        field_cmds, field_vars, mass_cmds, mass_vars = compile_projection(direction, dim, trailing, name, pbc=self._pbc)
        self._ensure_vars(field_vars | mass_vars)

        target = "n" * dim if direction == "c2n" else "c" * dim
        tmp = f"{name}_{'node' if target[0] == 'n' else 'cell'}"
        mass = f"{name}_{direction}_mass"

        self.zeros(tmp, target, trailing)
        for idx, rhs in field_cmds:
            self.add_to(tmp, idx, rhs)
        self.zeros(mass, target)
        for idx, rhs in mass_cmds:
            self.add_to(mass, idx, rhs)
        if trailing == (3,):
            self.add_line(f"{name} = {tmp} / {mass}.reshape({mass}.shape + (1,))")
        else:
            self.add_line(f"{name} = {tmp} / {mass}")

    def assign_sum(self, lhs, *terms, index=None):
        self.assign(lhs, self.sum(*terms), index)

    def retrn(self, code):
        self.add_line(f"return {code}")

    def retrn_sum(self, *terms):
        self.add_line(f"return {self.sum(*terms)}")

    # --- Backend-specific stubs (must be overridden) ---

    def zeros(self, name, spaces, shape=()):
        raise NotImplementedError

    def zeros_like(self, var, src, shape=None):
        raise NotImplementedError

    def assign(self, lhs, rhs, index=None):
        raise NotImplementedError

    def add_to(self, var, idx, rhs):
        raise NotImplementedError

    def retrn_expanded(self, code, shape):
        raise NotImplementedError

    def retrn_maximum(self, a, b):
        raise NotImplementedError


class CodeBlockBase(object):
    """
    Base class for backend-specific code blocks. Manages the string buffer
    of generated Python code.
    """

    # Subclasses must set this to the CodeFunction subclass they use.
    _code_function_class = None

    def __init__(self, preamble, pbc=None):
        self._code = preamble
        self._pbc = pbc

    def add_function(self, name, variables, var_spaces=None):
        return self._code_function_class(self, name, variables, var_spaces=var_spaces)

    def add(self, code):
        self._code += code

    def __str__(self):
        return self._code
