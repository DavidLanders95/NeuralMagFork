.. module:: neuralmag
    :noindex:

State, Mesh and Functions
=========================

The :class:`State` class plays a central role in every NeuralMag simulation. It's main purpose is to keep track of the current simulation state, such as the magnetization and the time.
Moreover, the :code:`state` object holds the mesh along with domain information and helps to connect the different solver modules through dynamic attributes.

Mesh and Functions
^^^^^^^^^^^^^^^^^^

The initialization of the :code:`state` requires a :code:`mesh` object.
As in standard finite-difference codes, the mesh in NeuralMag is a regular cuboid grid and is defined by the number of cells in the principal directions of the coordinate system and the size of the discretization cells.

.. code:: python

    # initialize mesh with 100 x 25 x 1 cells with cell-size 5 x 5 x 3 nm^3
    mesh = nm.Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
    state = nm.State(mesh)

After initializing the :code:`state`, it can be populated with attributes such as the time, material parameters or the magnetization.
Scalar values such as the time, can be simply set by

.. code:: python

    state.t = 1e-9

and are automatically converted to 0D PyTorch tensors by the :code:`state` object.
Fields, such as the magnetization, should be set as :class:`Function` objects, which contain information on the spatial discretization as well as the actual tensor data.
The :class:`Function` objects use the same type scheme for function spaces as the :class:`Variable <neuralmag.generators.pytorch_generator.Variable>` class used in NeuralMag's form compiler.
In order to initialize a vector function with nodal discretization for the magnetization, the :class:`Function` class is initialized as follows

.. code:: python

    state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
    m = nm.Function(state, "nnn", shape = (3,))

where the format string :code:`"nnn"` defines the nodal discretization for each of the 3 spatial dimensions.
Alternatively, one of the convenience wrappers can be used to initialize :class:`Function` objects with the most common function spaces

.. code:: python

    state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))

    # scalar nodal function
    f1 = nm.Function(state)
    print(f1.spaces, f1.shape)
    # results in "nnn ()"

    # vector nodal function
    f2 = nm.VectorFunction(state)
    print(f1.spaces, f1.shape)
    # results in "nnn (3,)"

    # scalar cell function
    f3 = nm.CellFunction(state)
    print(f1.spaces, f1.shape)
    # results in "ccc ()"

    # vector cell function
    f4 = nm.VectorCellFunction(state)
    print(f1.spaces, f1.shape)
    # results in "ccc (3,)"

Based on the :class:`Mesh` object of the :class:`State`, the function objects are initialized with PyTorch tensor of appropriate size.
For instance, a mesh with 100 x 25 x 1 cells as initialized above will result in a 4D tensor with shape 101 x 26 x 2 x 3 for a nodal vector function.

.. code:: python

    m.tensor.shape
    # torch.Size([101, 26, 2, 3])

In order to initialize a :class:`Function` with a constant value, :code:`fill` can be used.
For example, to initialize magnetization unit-vector field in z-direction

.. code:: python

    state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
    state.m = nm.VectorFunction(state).fill([0, 0, 1])

As described in :ref:`nodal_fd`, the magnetization is discretized on the nodes in nodal finite-differences, while the material parameters are discretized with cell functions.

.. code:: python

    state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
    state.material.Ms = nm.CellFunction(state).fill(8e5)

Here, we use the :code:`material` namespace within the :code:`state` object.

.. _dynamic attributes:

Dynamic Attributes
^^^^^^^^^^^^^^^^^^

The functionality of the :class:`State` class does not stop at the storage of static tensors and functions.
A central role of the :class:`State` class is the administration of functions and their dependencies to static tensors and other functions.
Dynamic attributes can be defined by functions that take PyTorch tensors as arguments and that return a PyTorch tensor.
When accessing dynamic attributes, the :class:`State` class does not return the function object, but instead it evaluates the function and returns the result.
If the function has arguments, the :class:`State` class does a lookup in it's own attributes with the argument names.
For example, we can easily define a dynamic attribute that returns a multiple of another attribute:

.. code:: python

    state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
    state.a = 1.0
    state.a2 = lambda a: 2*a
    print(state.a2)
    # results in "2.0"

Dynamic attributes can be arbitrarily chained

.. code:: python

    state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
    state.a4 = lambda a2: 2*a2
    print(state.a4)
    # results in "4.0"

On the first read access of a dynamic attribute, NeuralMag analyses the dependencies considering all involved dynamic attribues and dynamically compiles a Python function that only depends on static attributes of the state.
This function is cached along with the references to the static tensors that the function depends on, which leads to very fast compute times of the function for subsequent calls, especially if PyTorch's compile feature is enabled.

Internally, NeuralMag uses the feature of dynamic attributes for all numerical computations such as effective-field evaluations.
Registering an effective-field contribution with a state, effectively creates two dynamical attributes in the state, one for the effective field and one for the energy.
This architecture leads to a purely functional interface without any loops or conditional depending only on raw PyTorch tensors.
A functional interface is useful for efficient gradient computation and is indispensable for composable function transforms that we aim to support in later versions of NeuralMag via `functorch <https://pytorch.org/functorch/stable/>`_ or `JAX <https://jax.readthedocs.io/en/latest/quickstart.html>`_.

The Material Namespace
^^^^^^^^^^^^^^^^^^^^^^

In order to help the user to structure the attributes according to their role in the simulation, NeuralMag introduces a material namespace.
This namespace behaves exactly as the :class:`State` class itself with regard to the setting and getting of normal/dynamic attributes.
However, care has to be taken, when using members of the material namespace as arguments of dynamic attributes.
Since Python variables are not allowed to include the character "." in the name, NeuralMag expects the character sequence "__" instead as demonstrated in the following example.

.. code:: python

    # set up state and material.Ms
    state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
    state.material.Ms = nm.CellFunction(state).fill(8e5)

    # create dynamic attributes that computes Ms * m
    state.M = lambda m, material__Ms: m * material__Ms.unsqueeze(-1)

Class-Reference
^^^^^^^^^^^^^^^

.. autoclass:: Mesh
   :members:

.. autoclass:: State
   :members:

.. autoclass:: Function
   :members:

.. autoclass:: VectorFunction
   :members:

.. autoclass:: CellFunction
   :members:

.. autoclass:: VectorCellFunction
   :members:
