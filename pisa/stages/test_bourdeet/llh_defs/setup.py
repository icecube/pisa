from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
ext = Extension('poisson_gamma_mixtures', sources = ['poisson_gamma_mixtures.pyx', 'poisson_gamma.c'])
setup(name="PG_MIXTURES", ext_modules = cythonize([ext]),include_dirs=[numpy.get_include()])
