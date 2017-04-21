#!/usr/bin/env python
# authors: Sebastian Boeser, J.L. Lanfranchi, P. Eller, M. Hieronymus
"""
Allows for PISA installation. Tested with `pip`. Use the environment variable
`CC` to pass a custom compiler to the instller. (GCC and Clang should both
work; OpenMP support--an optional dependency--is only available for recent
versions of the latter).

Checkout the source code tree in the current directory via

    $ git clone https://github.com/jllanfranchi/pisa.git --branch cake \
        --single-branch

and install basic PISA package (in editable mode via -e flag) via

    $ pip install -e ./pisa -r ./pisa/requirements.txt

or include optional dependencies by specifying them in brackets

    $ pip install -e ./pisa[cuda,numba,develop] -r ./pisa/requirements.txt

If you wish to upgrade PISA and/or its dependencies:

    $ pip install ./pisa[cuda,numba,develop] -r ./pisa/requirements.txt \
        --upgrade

"""


from __future__ import absolute_import

from distutils.command.build import build
import os
import shutil
import subprocess
import sys
import tempfile

from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension, find_packages
import versioneer


# TODO: Compile CUDA kernel(s) here (since no need for dynamic install yet...
# unless e.g. datatype becomes optional and therefore compilation of the kernel
# needs to be done at run-time).

# TODO: address some/all of the following in the `setup()` method?
# * package_data
# * include_package_data
# * eager_resources


def setup_cc():
    """Set env var CC=cc if it is undefined"""
    if 'CC' not in os.environ or os.environ['CC'].strip() == '':
        os.environ['CC'] = 'cc'


def check_cuda():
    """pycuda is considered to be present if it can be imported"""
    try:
        import pycuda.driver
    except Exception:
        cuda = False
    else:
        cuda = True
    return cuda


# See http://openmp.org/wp/openmp-compilers/
OMP_TEST_PROGRAM = \
r"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}"""


def check_openmp():
    """OpenMP is present if a test program can compile with -fopenmp flag (e.g.
    some versions of Clang / gcc don't support OpenMP).

    Source: http://stackoverflow.com/questions/16549893

    """
    openmp = False
    setup_cc()
    tmpfname = r'test.c'
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    cc = os.environ['CC']
    try:
        with open(tmpfname, 'w', 0) as f:
            f.write(OMP_TEST_PROGRAM)
        with open(os.devnull, 'w') as fnull:
            returncode = subprocess.call([cc, '-fopenmp', tmpfname],
                                         stdout=fnull, stderr=fnull)
        # Successful build (possibly with warnings) means we can use OpenMP
        openmp = returncode == 0
    finally:
        # Restore directory location and clean up
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
    return openmp


class CustomBuild(build):
    """Define custom build order, so that the python interface module created
    by SWIG is staged in build_py.

    """
    # different order: build_ext *before* build_py
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts)
    ]


class CustomBuildExt(build_ext):
    """Replace default build_ext to allow for numpy install before setup.py
    needs it to include its dir.

    Code copied from coldfix / http://stackoverflow.com/a/21621689

    """
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def do_setup():
    """Perform the setup process"""
    setup_cc()
    sys.stdout.write('Using compiler %s\n' %os.environ['CC'])

    has_openmp = check_openmp()
    if not has_openmp:
        sys.stderr.write(
            'WARNING: Could not compile test program with -fopenmp;'
            ' installing PISA without OpenMP support.\n'
        )

    # Collect (build-able) external modules and package_data
    ext_modules = []
    package_data = {}

    # Include documentation files wherever they may be
    package_data[''] = ['*.md', '*.rst']

    # Prob3 oscillation code (pure C++, no CUDA)
    prob3cpu_module = Extension(
        name='pisa.stages.osc.prob3._BargerPropagator',
        sources=[
            'pisa/stages/osc/prob3/BargerPropagator.i',
            'pisa/stages/osc/prob3/BargerPropagator.cc',
            'pisa/stages/osc/prob3/EarthDensity.cc',
            'pisa/stages/osc/prob3/mosc.c',
            'pisa/stages/osc/prob3/mosc3.c'
        ],
        include_dirs=[
            'pisa/stages/osc/prob3/'
        ],
        extra_compile_args=['-Wall', '-O3', '-fPIC'],
        swig_opts=['-c++'],
    )
    ext_modules.append(prob3cpu_module)

    package_data['pisa.resources'] = [
        'aeff/*.json*',
        'cross_sections/*json*',
        'discr_sys/*.json*',

        'events/*.hdf5',
        'events/*.json*',
        'events/deepcore_ic86/MSU/1XXX/Joined/*.hdf5',
        'events/deepcore_ic86/MSU/1XXX/UnJoined/*.hdf5',
        'events/deepcore_ic86/MSU/1XXXX/Joined/*.hdf5',
        'events/deepcore_ic86/MSU/1XXXX/UnJoined/*.hdf5',
        'events/deepcore_ic86/MSU/icc/*.hdf5',
        'events/pingu_v36/*.hdf5',
        'events/pingu_v39/*.hdf5',

        'flux/*.d',
        'osc/*.hdf5',
        'osc/*.dat',
        'pid/*.json*',
        'priors/*.json*',
        'priors/*.md',
        'reco/*.json*',

        'settings/binning/*.cfg',
        'settings/discrete_sys/*.cfg',
        'settings/logging/logging.json',
        'settings/mc/*.cfg',
        'settings/minimizer/*.json*',
        'settings/osc/*.cfg',
        'settings/osc/*.md',
        'settings/pipeline/*.cfg',
        'settings/pipeline/*.md',

        'tests/data/aeff/*.json*',
        'tests/data/flux/*.json*',
        'tests/data/full/*.json*',
        'tests/data/osc/*.json*',
        'tests/data/pid/*.json*',
        'tests/data/reco/*.json*',
        'tests/data/xsec/*.root',
        'tests/data/oscfit/*.json*',
        'tests/settings/*.cfg'
    ]

    extra_compile_args = ['-O3', '-ffast-math', '-msse3',
                          '-ftree-vectorizer-verbose=1']
    extra_link_args = ['-ffast-math', '-msse2']
    if has_openmp:
        gaussians_cython_module = Extension(
            'pisa.utils.gaussians_cython',
            ['pisa/utils/gaussians_cython.pyx'],
            libraries=['m'],
            extra_compile_args=extra_compile_args + ['-fopenmp'],
            extra_link_args=extra_link_args + ['-fopenmp'],
        )
    else:
        gaussians_cython_module = Extension(
            'pisa.utils.gaussians_cython',
            ['pisa/utils/gaussians_cython.pyx'],
            libraries=['m'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ext_modules.append(gaussians_cython_module)

    cmdclasses = {'build': CustomBuild, 'build_ext': CustomBuildExt}
    cmdclasses.update(versioneer.get_cmdclass())

    # Now do the actual work
    setup(
        name='pisa',
        version=versioneer.get_version(),
        description='PINGU Simulation and Analysis',
        author='The IceCube/PINGU Collaboration',
        author_email='jll1062+pisa@phys.psu.edu',
        url='http://github.com/WIPACrepo/pisa',
        cmdclass=cmdclasses,
        python_requires='>=2.7',
        setup_requires=[
            'pip>=1.8',
            'setuptools>18.5', # versioneer requires >18.5
            'cython',
            'numpy>=1.11',
        ],
        install_requires=[
            'scipy>=0.17',
            'dill',
            'h5py',
            'line_profiler',
            'matplotlib>=1.5', # inferno colormap
            'pint>=0.8', # earlier versions buggy
            'kde',
            'simplejson>=3.2',
            'tables',
            'uncertainties'
        ],
        extras_require={
            'cuda': [
                'pycuda'
            ],
            'numba': [
                'llvmlite>=0.16', # fastmath jit flag
                'numba>=0.31' # fastmath jit flag
            ],
            'develop': [
                'pylint>=1.7',
                'recommonmark',
                'sphinx>=1.3',
                'sphinx_rtd_theme',
                'versioneer',
            ]
        },
        packages=find_packages(),
        ext_modules=ext_modules,
        package_data=package_data,
        # Cannot be compressed due to c, pyx, and cu source files that need to
        # be compiled and are inaccessible in zip
        zip_safe=False,
        entry_points={
            'console_scripts': [
                # Scripts in analysis dir
                'hypo_testing.py = pisa.analysis.hypo_testing:main',
                'hypo_testing_postprocess.py = pisa.analysis.hypo_testing_postprocess:main',
                'profile_llh_analysis.py = pisa.analysis.profile_llh_analysis:main',
                'profile_llh_postprocess.py = pisa.analysis.profile_llh_postprocess:main',

                # Scripts in core dir
                'distribution_maker.py = pisa.core.distribution_maker:main',
                'pipeline.py = pisa.core.pipeline:main',

                # Scripts in scripts dir
                'add_flux_to_events_file.py = pisa.scripts.add_flux_to_events_file:main',
                'compare.py = pisa.scripts.compare:main',
                'fit_discrete_sys.py = pisa.scripts.fit_discrete_sys:main',
                'make_asymmetry_plots.py = pisa.scripts.make_asymmetry_plots:main',
                'make_events_file.py = pisa.scripts.make_events_file:main',
                'make_nufit_theta23_spline_priors.py = pisa.scripts.make_nufit_theta23_spline_priors:main',
            ]
        }
    )
    if not check_cuda():
        sys.stderr.write('WARNING: Could not import pycuda; attempt will be '
                         ' made to install, but if this fails, PISA may not be'
                         ' able to support CUDA (GPU) accelerations.\n')


if __name__ == '__main__':
    do_setup()
