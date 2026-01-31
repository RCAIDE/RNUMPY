# __init__.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Aug 2024 E. Botero
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  Package Imports
# ----------------------------------------------------------------------------------------------------------------------  

import warnings
import numpy as np
import scipy as sp
try:
    import jax as jax
    from jax import Array as jarray
except ImportError:
    warnings.warn("The optional package, JAX is not installed. Autograd and JIT are unavailable", ImportWarning)
    jax = None
    jarray = None

# Set the default environment
use_jax      = False
jax_handle   = jax
numpy_handle = np
scipy_handle = sp

# Set pi
pi = 3.141592653589793


# ----------------------------------------------------------------------------------------------------------------------
#  Basic Array Stuff
# ----------------------------------------------------------------------------------------------------------------------  

from ._basearrays import _set_array_base_attributes

class Array():
    pass

class JaxArray(Array,jarray):
     pass

class NumpyArray(Array,np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        # Convert input_array to an instance of MyArray
        obj = np.asarray(input_array).view(cls)
        return obj

# Dynamically register JAX-style methods on the arrays
_set_array_base_attributes(NumpyArray, exclude={'__getitem__'})


# ----------------------------------------------------------------------------------------------------------------------
# Project Imports
# ----------------------------------------------------------------------------------------------------------------------  

# Finally import scripts
from .src import *
from .linalg import *
from .lax import *
from . import scipy


