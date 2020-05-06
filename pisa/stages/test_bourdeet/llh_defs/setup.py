from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
ext1 = Extension('llh_fast', sources = ['llh_fast.pyx', 'llh_fast_defs.c', 'poisson_gamma.c'])
ext2 = Extension('poisson_gamma_mixtures', sources = ['poisson_gamma_mixtures.pyx', 'poisson_gamma.c'])

setup(name="LLHFAST", ext_modules = cythonize([ext1]),include_dirs=[numpy.get_include()])
setup(name="PG_MIXTURES", ext_modules = cythonize([ext2]),include_dirs=[numpy.get_include()])
