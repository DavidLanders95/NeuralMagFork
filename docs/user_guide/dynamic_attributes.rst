Dynamic Attributes, ``resolve`` and ``remap``
=============================================

The single feature that makes inverse problems comfortable in NeuralMag is
that almost everything you put on a :class:`~neuralmag.State` can be a plain
Python lambda of other state attributes. This page is split in two halves:
the first explains the everyday use of dynamic attributes; the second shows
how :meth:`State.resolve` and :meth:`State.remap` turn that machinery into
pure functions you can hand to ``jax.grad`` or ``torch.autograd``.

Part A — Everyday dynamic attributes
------------------------------------

Setting and getting state attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you write ``state.X = something`` NeuralMag does *not* attach
``something`` directly to the Python object. The :class:`~neuralmag.State`
class overrides ``__setattr__`` and ``__getattr__`` and stores values in
three dictionaries:

* ``state._attr_values`` — the raw value (a tensor, a constant, or a
  callable).
* ``state._attr_types`` — for attributes that should always be wrapped as a
  :class:`~neuralmag.Function` (the field-term outputs do this when they
  register).
* ``state._attr_funcs`` — a cache of *resolved* callables for attributes that
  were assigned a lambda. The cache is cleared whenever you reassign any
  attribute.

The conversion rules in ``__setattr__`` are simple:

* an ``int`` or ``float`` is converted to a backend tensor automatically;
* a ``list`` is converted to a tensor when possible;
* a callable is stored as-is and triggers a cache clear;
* anything else (e.g. a pre-built ``Function`` or backend tensor) is stored
  as-is.

Reading the attribute back via ``state.X`` does the inverse: if the stored
value is callable, NeuralMag :meth:`resolve <State.resolve>` it the first
time, inspects the resolved function's signature, fetches the named
arguments from the state itself, and finally calls the function. The
resolved function is memoized in ``_attr_funcs``.

Accepted right-hand sides
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import neuralmag as nm

   mesh  = nm.Mesh((10, 10, 1), (5e-9, 5e-9, 3e-9))
   state = nm.State(mesh)

   # 1) plain scalar -> auto-converted to a backend tensor
   state.T = 200.0

   # 2) pre-built Function or CellFunction
   state.material.Ms = nm.CellFunction(state).fill(8e5)

   # 3) lambda depending on other state attributes
   Ms0, Tc = 8e5, 400.0
   state.material.Ms = lambda T: Ms0 * (1 - T / Tc) ** 1.5

   # 4) lambda returning a tensor for a Function-typed attribute
   state.m = nm.VectorFunction(
       state, tensor=lambda T: state.tensor([0.0, 0.0, 1.0 - T / Tc])
   )

In case (3), reading ``state.material.Ms`` evaluates the lambda lazily on
first access and returns a tensor; reassigning ``state.T`` invalidates the
resolved-function cache so the next read picks up the new temperature.

The material namespace
^^^^^^^^^^^^^^^^^^^^^^

``state.material`` is a tiny proxy object: ``state.material.Ms`` is shorthand
for ``state.material__Ms``. The double underscore is a deliberate
name-mangling so material parameters share the same dynamic-attribute
machinery as everything else on the state — there is no second store, no
separate dependency graph.

Lazy evaluation and caching
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two facts are useful to keep in mind:

1. Reading ``state.X`` for a lambda-valued attribute *executes* the lambda
   every time. The "cache" only memoizes the *resolved* call graph, not its
   numeric output. If a lambda is expensive, wrap its result in your own
   ``Function`` or compute it once outside the loop.
2. Any reassignment to a state attribute clears ``_attr_funcs`` entirely. In
   inner optimization loops you typically *do not* want to reassign
   structural attributes — only the design tensor (see Part B).

.. admonition:: Gotchas
   :class: warning

   * **Argument names matter.** A lambda's parameter names are matched
     against existing state attribute names. ``lambda temperature: ...``
     will fail if you assigned the temperature as ``state.T``.
   * **Don't capture tensors by closure.** If a tensor is captured implicitly
     (``Ms = state.material.Ms.tensor; lambda x: Ms * x``), neither
     :meth:`resolve <State.resolve>` nor :meth:`remap <State.remap>` can see
     it. Pass it through the parameter list instead so it shows up in the
     dependency graph.
   * **Plain tensor assignment bypasses the lambda path.** Setting
     ``state.material.Ms = tensor_value`` overwrites a previously assigned
     lambda. That is the correct behaviour but easy to do by accident.

Part B — ``resolve`` and ``remap`` for inverse problems
-------------------------------------------------------

When you write an optimization loop you typically need a function whose
*only* arguments are the design variables — everything else (mesh, material
parameters that you are not optimizing, the magnetization that you are
relaxing inside the forward solve) should be pre-bound. That is exactly what
:meth:`State.resolve` does.

``state.resolve``
^^^^^^^^^^^^^^^^^

Signature::

   state.resolve(func, func_args=None, remap={}, inject={})

* ``func`` — a callable, or the *name* of a state attribute (e.g. ``"h_demag"``).
* ``func_args`` — the names of the arguments the returned function should
  expose. Anything not listed here is bound from the current state.
* ``remap`` — rename arguments along the way; recursively applied to
  dependencies.
* ``inject`` — replace named dependencies by user-supplied callables (handy
  for testing or for swapping a sub-model).

Internally :meth:`resolve <State.resolve>` walks the dependency graph
(``_collect_func_deps``) and emits a small Python function that wires
sub-functions together with the right argument order, then snapshots the
remaining bound values into its globals. The result is a regular Python
callable — there is no NeuralMag-specific glue at call time, which is what
makes it differentiable through ``jax.grad`` / ``torch.autograd``.

A few minimal examples (mirroring ``tests/unit/state_test.py``):

.. code-block:: python

   # Walking a chain: c depends on b, b depends on a
   state.a = 1.0
   state.b = lambda a: 2.0 * a
   c = lambda b: 2.0 * b
   func = state.resolve(c)          # signature: func(a)
   assert func(1.0) == 4.0          # because b = 2*a, c = 2*b

.. code-block:: python

   # Pre-binding everything except the design variable
   state.a = 2.0
   state.b = 4.0
   c = lambda a, b: a * b
   func = state.resolve(c, ["a"])   # b is bound to 4.0 from state
   assert func(1.0) == 4.0

.. code-block:: python

   # Swapping a sub-dependency on the fly
   state.a = 2.0
   state.b = 4.0
   c = lambda a, b: a * b
   func = state.resolve(c, ["e"], inject={"b": lambda e: 2 * e})
   assert func(1.0) == 4.0          # c(a, b(e)) = 2 * (2*1) = 4

The ``resolve`` call is the cornerstone of the topology-optimization demo:

.. code-block:: python

   # demos/topology-optimization_jax.py
   demag_func = state.resolve("h_demag", ["rho_m"])
   #          ^                           ^
   #          attribute name              the only free argument

After this line ``demag_func(rho_m_tensor)`` returns the demagnetization
field for an arbitrary design tensor with everything else (mesh, ``Ms``,
relaxed magnetization, …) baked in.

``state.remap``
^^^^^^^^^^^^^^^

:meth:`State.remap` is the much smaller cousin: it just renames a function's
arguments and otherwise leaves everything alone.

.. code-block:: python

   def f(a, b):
       return a + b

   g = state.remap(f, {"a": "x", "b": "y"})
   # g(x, y) is now identical to f(a, b)

The most common real-world use is internal: when an :class:`ExternalField` is
registered with a custom name (so it shows up as ``state.h_my_ext`` instead
of ``state.h_external``), the field term remaps the energy function's
``h_external`` argument to the new name so that
:meth:`resolve <State.resolve>` finds it under the right key.

A typical inverse-problem loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Putting it together, an optimization loop in JAX looks like this:

.. code-block:: python

   import jax, jax.numpy as jnp
   import neuralmag as nm

   state.rho_m = nm.CellFunction(state).fill(1.0)
   state.rho   = nm.CellFunction(
       state, tensor=lambda rho_m: jnp.where(mask, rho_m, state.eps)
   )

   demag_func = state.resolve("h_demag", ["rho_m"])

   def loss(rho_m):
       h = demag_func(rho_m ** 3)
       return -(h[10, 10, 12, 2] ** 2)

   grad_loss = jax.grad(loss)

   for step in range(N):
       g = grad_loss(state.rho_m.tensor)
       state.rho_m.tensor = jnp.clip(
           state.rho_m.tensor - lr * g, state.eps, 1.0
       )

Two things are worth noting:

* ``state.resolve`` is called **once, outside the loop**. Its compile cost
  is non-trivial and the resolved closure remains valid as long as you only
  modify the *tensor* of ``state.rho_m`` (which you do via ``.tensor =
  ...``), not the lambda graph itself.
* The rest of the loop is plain JAX (or PyTorch). NeuralMag is just the
  thing that produced the differentiable ``demag_func``.

See also
--------

* :doc:`domains` — how ``rho``, ``add_domain`` and ``fill_by_domain``
  interact with the dynamic-attribute machinery.
* :doc:`discretization` — what ``state.h_*`` actually computes under the
  hood.
