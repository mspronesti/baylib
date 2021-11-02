"""
Setup file for pybind11 integration with baylib
Adapted from online sources, inspired by a couple of
github repositories (e.g. googlebenchmark).
Run
  python3 setup.py build
to build it or
  python3 setup.py install
to install the pyhton integration for baylib
"""

from subprocess import CalledProcessError
import os
import re
import sys
import platform
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[], extra_compile_args=['-std=c++20', '-Wall'])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def get_cmake_version(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except:
            sys.stderr.write("\nERROR: CMake must be installed to build baylib\n\n")
            sys.exit(1)
        return re.search(r"version\s*([\d.]+)", out.decode()).group(1)

    def run(self):
        cmake_version = self.get_cmake_version()
        if platform.system() == "Windows":
            if LooseVersion(cmake_version) < '3.1.0':
                sys.stderr.write("\nERROR: CMake >= 3.1.0 is required on Windows\n\n")
                sys.exit(1)

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        build_folder = os.path.abspath(self.build_temp)
        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

        cmake_setup = ['cmake', ext.sourcedir] + cmake_args
        cmake_build = ['cmake', '--build', '.'] + build_args

        print("Building extension for Python {}".format(sys.version.split('\n', 1)[0]))
        print("Invoking CMake setup: '{}'".format(' '.join(cmake_setup)))
        sys.stdout.flush()
        subprocess.check_call(cmake_setup, cwd=build_folder)
        print("Invoking CMake build: '{}'".format(' '.join(cmake_build)))
        sys.stdout.flush()
        subprocess.check_call(cmake_build, cwd=build_folder)


kwargs = dict(
    name='baylib',
    version='0.0.1',
    author='Massimiliano Pronesti',
    author_email='massimiliano.pronesti@gmail.com',
    description='A parallel inference library for discrete bayesian networks',
    long_description='',
    ext_modules=[CMakeExtension('_pybaylib', 'python')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=['baylib'],
    package_dir={'': 'python'},
)

try:
    setup(**kwargs)
except CalledProcessError:
    print('Failed to build extension!')
    del kwargs['ext_modules']
    setup(**kwargs)


