.. module:: neuralmag
    :noindex:

Field Terms
===========

Effective-field contributions are subclasses of :class:`FieldTerm`. Calling
``term.register(state)`` adds two dynamic attributes to the state:
``state.h_<name>`` for the effective field and ``state.E_<name>`` for the
energy. Multiple terms can be combined through :class:`TotalField`.

.. seealso::

   * :doc:`../user_guide/discretization` — nodal finite-difference and
     FIC schemes that underlie every field term.
   * :doc:`../user_guide/form_compiler` — how ``e_expr`` is turned into
     vectorized backend code.
   * :doc:`../user_guide/dynamic_attributes` — what registering a field
     term actually does to the :class:`State`.

.. autoclass:: FieldTerm
   :members:

.. autoclass:: TotalField
   :members:

.. _predefined field terms:

Predefined Field Terms
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BulkDMIField

.. autoclass:: DemagField

.. autoclass:: CubicAnisotropyField

.. autoclass:: ExchangeField

.. autoclass:: ExternalField

.. autoclass:: InterfaceDMIField

.. autoclass:: InterlayerExchangeField

.. autoclass:: UniaxialAnisotropyField
