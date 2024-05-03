.. module:: neuralmag
    :noindex:

Field Terms
===========

NeuralMag supports a number of standard micromagnetic energy contributions through subclasses of the :class:`FieldTerm` class, see :ref:`predefined field terms`.
This base class not only acts as an abstract class defining the interface of a field term.
Moreover, it provides a powerful just-in-time compiler that is able to turn the symbolic description of an energy contribution into efficient code for the computation of the energy and corresponding effective field.
To this end, it uses symbolic differentiation and integration to implement the respective effective-field contribution using :ref:`nodal_fd`.
For more complicated field terms which require taylored numerical algorithms, such as the :class:`DemagField`, the standard methods of the :class:`FieldTerm` class can be overwritten.

In order to use field terms, they have to be registered with a :class:`State` object.
By registering the field term, the state is extended by :ref:`dynamic attributes` for the evaluation of the energy and the effective field.
In order to combine multiple field contributions to the total effective field, NeuralMag provides the :class:`TotalField` class that automatically adds up arbitrary field contributions and energies.

.. autoclass:: FieldTerm
   :members:

.. autoclass:: TotalField
   :members:

.. _predefined field terms:

Predefined Field Terms
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BulkDMIField
   :members:

.. autoclass:: DemagField
   :members:

.. autoclass:: ExchangeField
   :members:

.. autoclass:: InterfaceDMIField
   :members:

.. autoclass:: InterlayerExchangeField
   :members:

.. autoclass:: UniaxialAnisotropyField
   :members:
