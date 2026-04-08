.. module:: neuralmag
    :noindex:

State, Mesh and Functions
=========================

API reference for the :class:`Mesh`, :class:`State`, and :class:`Function`
classes.

.. seealso::

   * :doc:`../user_guide/introduction` — mental model and how the state
     fits into a simulation.
   * :doc:`../user_guide/dynamic_attributes` — how ``state.X = lambda …``
     and :meth:`State.resolve` / :meth:`State.remap` work.
   * :doc:`../user_guide/domains` — ``state.rho``, :meth:`State.add_domain`
     and :meth:`Function.fill_by_domain`.

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
