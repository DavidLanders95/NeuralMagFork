#!/usr/bin/env python

from distutils.core import setup

setup(name='magnum.nf',
      version='0.0.1',
      description='magnum.nf is a micromagnetic GPU code implementing the nodal FD discretization',
      author='Claas Abert',
      author_email='claas.abert@univie.ac.at',
      packages=['magnumnf', 'magnumnf.common', 'magnumnf.field_terms', 'magnumnf.solvers'],
      #package_data = {},
      #entry_points={},
      install_requires=[
          'numpy',
          'pyvista'
          'sympy',
          'scipy',
          'torch',
          'numba'
      ]
     )
