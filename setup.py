from distutils.core import setup
from Cython.Build import cythonize
import numpy, math

setup(
  # name = 'Hello world app',
  ext_modules = cythonize("*.pyx", compiler_directives={"boundscheck": False, "cdivision": True, "wraparound": False, "nonecheck": False}),
  include_dirs=[numpy.get_include()]
)