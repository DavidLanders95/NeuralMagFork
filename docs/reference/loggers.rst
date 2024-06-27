.. module:: neuralmag
    :noindex:

Loggers
=======

NeuralMag provides several helper classes for the logging of simulation results.
These classes mainly differ in the kind of data that is logged.
While the :class:`FieldLogger` class is able to write VTK files for arbitrary field data such as the magnetization field or material parameters, the :class:`ScalarLogger` class writes scalar data such as the averaged magnetization to as CSV file.
The :class:`Logger` class offers both scalar and field logging and has the additional functionality to resume simulations from existing log files.

Class-Reference
^^^^^^^^^^^^^^^

.. autoclass:: FieldLogger
   :members:

.. autoclass:: ScalarLogger
   :members:

.. autoclass:: Logger
   :members:
