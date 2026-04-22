# __init__.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Aug 2024 E. Botero
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  Package Imports
# ----------------------------------------------------------------------------------------------------------------------  

import numpy as np
import builtins
import warnings
import inspect

# Set the handles to None initially to allow submodules to import RNUMPY
jax_handle   = None
numpy_handle = np
scipy_handle = None
torch_handle = None

try:
    import scipy as sp
    scipy_handle = sp
except ImportError:
    warnings.warn("Scipy is not installed.", ImportWarning)

try:
    import jax
    from jax import Array as jarray
    jax_handle = jax
except ImportError:
    warnings.warn("The optional package, JAX is not installed. Autograd and JIT are unavailable", ImportWarning)
    jax = None
    jarray = None

try:
    import torch
    from torch import Tensor as ttensor
    torch_handle = torch
except ImportError:
    warnings.warn("The optional package, PyTorch is not installed. Torch backend is unavailable", ImportWarning)
    torch = None
    ttensor = None

# Set the default environment
use_jax      = False
use_torch    = False
ensure_differentiable = True

# Set pi
pi = 3.141592653589793

# x64 toggle
x64_enabled = False

def enable_x64(enabled=True):
    global x64_enabled
    x64_enabled = enabled
    if torch_handle is not None:
        torch_handle.set_default_dtype(torch_handle.float64 if enabled else torch_handle.float32)
    if jax_handle is not None:
        jax_handle.config.update("jax_enable_x64", enabled)

# For JAX this is straight numpy
from numpy import inf, newaxis, nan

# Data Types
# These are handled dynamically via __getattr__ for backend compatibility
_DTYPE_MAP = {
    'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'double': 'float64',
    'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64',
    'uint8': 'uint8', 'uint16': 'uint16', 'uint32': 'uint32', 'uint64': 'uint64',
    'bool_': 'bool', 'complex64': 'complex64', 'complex128': 'complex128'
}

_NUMPY_DTYPE_FALLBACKS = {
    'float16': np.float16, 'float32': np.float32, 'float64': np.float64, 'double': np.double,
    'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
    'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
    'bool_': np.bool_, 'complex64': np.complex64, 'complex128': np.complex128
}

def __getattr__(name):
    if name in _DTYPE_MAP:
        if use_torch and torch_handle:
            tr_name = _DTYPE_MAP[name]
            if builtins.hasattr(torch_handle, tr_name):
                 return builtins.getattr(torch_handle, tr_name)
        elif use_jax and jax_handle:
             # Jax reuses numpy dtypes effectively
             return _NUMPY_DTYPE_FALLBACKS[name]
        return _NUMPY_DTYPE_FALLBACKS[name]
    
    if name == 'float': return builtins.float
    if name == 'int': return builtins.int
    if name == 'bool': return builtins.bool
    if name == 'complex': return builtins.complex

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
     return sorted(set(globals().keys()) | set(_DTYPE_MAP.keys()) | {'int', 'float', 'bool', 'complex'})

# ----------------------------------------------------------------------------------------------------------------------
#  Basic Array Stuff
# ----------------------------------------------------------------------------------------------------------------------  

from ._basearrays import _set_array_base_attributes

def _get_obj_name(obj):
    try:
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name in ('__setitem__', '_inplace_error', '_get_obj_name', '<lambda>') or \
               frame.f_code.co_filename.endswith('RNUMPY/__init__.py'):
                frame = frame.f_back
                continue
            
            for name, val in frame.f_locals.items():
                if val is obj and name != 'self':
                    return name
            for name, val in frame.f_globals.items():
                if val is obj:
                    return name
            frame = frame.f_back
    except:
        pass
    return "x"

class Array():
    pass

class JaxArray(Array,jarray):
     pass

class NumpyArray(Array,np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        # Convert input_array to an instance of MyArray
        obj = np.asarray(input_array).view(cls)
        return obj

    def __setitem__(self, key, value):
        if not ensure_differentiable:
            return super().__setitem__(key, value)
        name = _get_obj_name(self)
        
        # Detect if key is likely a boolean mask
        is_bool = False
        if isinstance(key, (bool, np.bool_)):
            is_bool = True
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            is_bool = True
        elif torch_handle is not None and isinstance(key, torch_handle.Tensor) and key.dtype == torch_handle.bool:
            is_bool = True
        elif isinstance(key, list) and len(key) > 0 and isinstance(key[0], bool):
            is_bool = True

        if is_bool:
             key_name = _get_obj_name(key)
             if key_name == "x": key_name = str(key)
             raise TypeError(
                f"RNUMPY: Inplace assignment {name}[{key_name}] = value with a boolean mask is not allowed in differentiable code. "
                f"Please use {name} = rp.where({key_name}, value, {name}) instead."
            )
        else:
            raise TypeError(
                f"RNUMPY: Inplace assignment {name}[{key}] = value is not allowed in differentiable code. "
                f"Please use {name} = {name}.at[{key}].set(value) instead."
            )

    def __iadd__(self, other):
        if not ensure_differentiable: return super().__iadd__(other)
        name = _get_obj_name(self)
        raise TypeError(f"RNUMPY: Inplace operator {name} += value is not allowed in differentiable code. "
                        f"Please use {name} = {name} + value or {name}.at[...].add(value) instead.")

    def __isub__(self, other):
        if not ensure_differentiable: return super().__isub__(other)
        name = _get_obj_name(self)
        raise TypeError(f"RNUMPY: Inplace operator {name} -= value is not allowed in differentiable code. "
                        f"Please use {name} = {name} - value or {name}.at[...].subtract(value) instead.")

    def __imul__(self, other):
        if not ensure_differentiable: return super().__imul__(other)
        name = _get_obj_name(self)
        raise TypeError(f"RNUMPY: Inplace operator {name} *= value is not allowed in differentiable code. "
                        f"Please use {name} = {name} * value or {name}.at[...].multiply(value) instead.")

    def __itruediv__(self, other):
        if not ensure_differentiable: return super().__itruediv__(other)
        name = _get_obj_name(self)
        raise TypeError(f"RNUMPY: Inplace operator {name} /= value is not allowed in differentiable code. "
                        f"Please use {name} = {name} / value or {name}.at[...].divide(value) instead.")

    def __ipow__(self, other):
        if not ensure_differentiable: return super().__ipow__(other)
        name = _get_obj_name(self)
        raise TypeError(f"RNUMPY: Inplace operator {name} **= value is not allowed in differentiable code. "
                        f"Please use {name} = {name} ** value or {name}.at[...].power(value) instead.")

if torch is not None:
    class TorchArray(Array, ttensor):
        def __new__(cls, x, *args, **kwargs):
            return torch.as_tensor(x, *args, **kwargs).as_subclass(cls)

        def __getitem__(self, index):
            # NumPy allows slicing 0-d arrays: scalar[:] -> scalar, scalar[:, None] -> (1, 1) scalar
            # PyTorch raises IndexError: too many indices for tensor of dimension 0
            if self.dim() == 0:
                if isinstance(index, slice) and index == builtins.slice(None):
                    return self
                elif isinstance(index, tuple):
                    # Check if all elements are either None (newaxis) or a full slice [:]
                    if builtins.all(i is None or (isinstance(i, builtins.slice) and i == builtins.slice(None)) for i in index):
                        res = self
                        for i in index:
                            if i is None:
                                res = res.unsqueeze(0)
                        return res.as_subclass(TorchArray)
            
            # Handle negative steps if torch version is old or for subclass compatibility
            # We check if any slice in the index has a negative step.
            has_neg_step = False
            if isinstance(index, slice):
                if index.step is not None and index.step < 0:
                    has_neg_step = True
            elif isinstance(index, tuple):
                if builtins.any(isinstance(i, slice) and getattr(i, 'step', None) is not None and i.step < 0 for i in index):
                    has_neg_step = True

            if has_neg_step:
                # Fallback: PyTorch slicing with negative steps can be restrictive depending on version/subclass.
                # For basic 1D reverse [::-1], use flip.
                if isinstance(index, slice) and index == builtins.slice(None, None, -1) and self.dim() == 1:
                     return torch.flip(self, [0]).as_subclass(TorchArray)
                
                # General case fallback to numpy for complex negative step slicing
                np_res = self.detach().cpu().numpy()[index]
                # PyTorch cannot convert numpy arrays with negative strides directly, so always copy
                return TorchArray(np_res.copy()).to(self.device).as_subclass(TorchArray)

            res = super().__getitem__(index)
            if isinstance(res, ttensor) and not isinstance(res, TorchArray):
                return res.as_subclass(TorchArray)
            return res

        def __setitem__(self, key, value):
            if not ensure_differentiable:
                return super().__setitem__(key, value)
            name = _get_obj_name(self)
            
            # Detect if key is likely a boolean mask
            is_bool = False
            if isinstance(key, (bool, np.bool_)):
                is_bool = True
            elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
                is_bool = True
            elif torch_handle is not None and isinstance(key, torch_handle.Tensor) and key.dtype == torch_handle.bool:
                is_bool = True
            elif isinstance(key, list) and len(key) > 0 and isinstance(key[0], bool):
                is_bool = True

            if is_bool:
                 key_name = _get_obj_name(key)
                 if key_name == "x": key_name = str(key)
                 raise TypeError(
                    f"RNUMPY: Inplace assignment {name}[{key_name}] = value with a boolean mask is not allowed in differentiable code. "
                    f"Please use {name} = rp.where({key_name}, value, {name}) instead."
                )
            else:
                raise TypeError(
                    f"RNUMPY: Inplace assignment {name}[{key}] = value is not allowed in differentiable code. "
                    f"Please use {name} = {name}.at[{key}].set(value) instead."
                )

        def __iadd__(self, other):
            if not ensure_differentiable: return super().__iadd__(other)
            name = _get_obj_name(self)
            raise TypeError(f"RNUMPY: Inplace operator {name} += value is not allowed in differentiable code. "
                            f"Please use {name} = {name} + value or {name}.at[...].add(value) instead.")

        def __isub__(self, other):
            if not ensure_differentiable: return super().__isub__(other)
            name = _get_obj_name(self)
            raise TypeError(f"RNUMPY: Inplace operator {name} -= value is not allowed in differentiable code. "
                            f"Please use {name} = {name} - value or {name}.at[...].subtract(value) instead.")

        def __imul__(self, other):
            if not ensure_differentiable: return super().__imul__(other)
            name = _get_obj_name(self)
            raise TypeError(f"RNUMPY: Inplace operator {name} *= value is not allowed in differentiable code. "
                            f"Please use {name} = {name} * value or {name}.at[...].multiply(value) instead.")

        def __itruediv__(self, other):
            if not ensure_differentiable: return super().__itruediv__(other)
            name = _get_obj_name(self)
            raise TypeError(f"RNUMPY: Inplace operator {name} /= value is not allowed in differentiable code. "
                            f"Please use {name} = {name} / value or {name}.at[...].divide(value) instead.")

        def __ipow__(self, other):
            if not ensure_differentiable: return super().__ipow__(other)
            name = _get_obj_name(self)
            raise TypeError(f"RNUMPY: Inplace operator {name} **= value is not allowed in differentiable code. "
                            f"Please use {name} = {name} ** value or {name}.at[...].power(value) instead.")

        def __deepcopy__(self, memo):
            return TorchArray(self.clone())

        @property
        def size(self):
            # NumPy uses .size as an attribute (int), but PyTorch uses .size() as a method (returns torch.Size).
            # To support both, we return a subclass of int that also implements __call__ to return the shape.
            class SizeInt(int):
                def __call__(self, *args, **kwargs):
                    return ttensor.size(self._obj, *args, **kwargs)
            res = SizeInt(self.numel())
            res._obj = self
            return res

        def new_empty(self, size, dtype=None, device=None, requires_grad=False):
            return ttensor.new_empty(self, size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(TorchArray)

        # Reductions to match NumPy signature
        def max(self, axis=None, out=None, keepdims=False, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is None:
                res = super().max()
                if out is not None: out.copy_(res)
                if isinstance(res, ttensor) and not isinstance(res, TorchArray):
                    return res.as_subclass(TorchArray)
                return res
            res_tuple = torch.max(self, dim=dim, keepdim=bool(keepdims), out=out)
            return TorchArray(res_tuple[0] if out is not None else res_tuple[0])

        def min(self, axis=None, out=None, keepdims=False, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is None:
                res = super().min()
                if out is not None: out.copy_(res)
                if isinstance(res, ttensor) and not isinstance(res, TorchArray):
                    return res.as_subclass(TorchArray)
                return res
            res_tuple = torch.min(self, dim=dim, keepdim=bool(keepdims), out=out)
            return TorchArray(res_tuple[0] if out is not None else res_tuple[0])

        def sum(self, axis=None, out=None, keepdims=False, dtype=None, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is not None:
                return TorchArray(torch.sum(self, dim=dim, keepdim=bool(keepdims), dtype=dtype, out=out))
            
            # Global sum does not support keepdim in all torch versions
            res = torch.sum(self, dtype=dtype)
            if keepdims:
                for _ in range(self.dim()):
                    res = res.unsqueeze(0)
            
            if out is not None:
                out.copy_(res)
                
            return TorchArray(res)

        def mean(self, axis=None, out=None, keepdims=False, dtype=None, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is not None:
                 return TorchArray(torch.mean(self, dim=dim, keepdim=bool(keepdims), dtype=dtype, out=out))
            return TorchArray(torch.mean(self, keepdim=bool(keepdims), dtype=dtype, out=out))

        def std(self, axis=None, out=None, keepdims=False, unbiased=True, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is not None:
                 return TorchArray(torch.std(self, dim=dim, keepdim=bool(keepdims), unbiased=unbiased, out=out))
            return TorchArray(torch.std(self, keepdim=bool(keepdims), unbiased=unbiased, out=out))

        def var(self, axis=None, out=None, keepdims=False, unbiased=True, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is not None:
                 return TorchArray(torch.var(self, dim=dim, keepdim=bool(keepdims), unbiased=unbiased, out=out))
            return TorchArray(torch.var(self, keepdim=bool(keepdims), unbiased=unbiased, out=out))

        def all(self, axis=None, out=None, keepdims=False, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is not None:
                 return TorchArray(torch.all(self, dim=dim, keepdim=bool(keepdims), out=out))
            return TorchArray(torch.all(self))

        def any(self, axis=None, out=None, keepdims=False, **kwargs):
            dim = axis if axis is not None else kwargs.get('dim', None)
            if dim is not None:
                 return TorchArray(torch.any(self, dim=dim, keepdim=bool(keepdims), out=out))
            return TorchArray(torch.any(self))

        def ravel(self, order='C'):
            if order == 'C':
                return TorchArray(self.reshape(-1))
            elif order == 'F':
                return TorchArray(self.permute(tuple(reversed(range(self.dim())))).reshape(-1))
            elif order == 'A' or order == 'K':
                return TorchArray(self.reshape(-1))
            else:
                raise ValueError(f"order '{order}' not understood")

        def flatten(self, order='C'):
            return self.ravel(order=order)

        def __len__(self):
            if self.dim() == 0:
                return 1
            return super().__len__()

        def __array_wrap__(self, array, context=None):
            if isinstance(array, np.ndarray) and array.dtype == np.dtype('O'):
                return NumpyArray(array)
            res = super().__array_wrap__(array, context)
            if isinstance(res, ttensor) and not isinstance(res, TorchArray):
                return res.as_subclass(TorchArray)
            return res
else:
    class TorchArray(Array):
        pass

# Dynamically register JAX-style methods on the arrays
_set_array_base_attributes(NumpyArray, exclude={'__getitem__'})
_set_array_base_attributes(TorchArray, exclude={'__getitem__'})


# ----------------------------------------------------------------------------------------------------------------------
# Project Imports
# ----------------------------------------------------------------------------------------------------------------------  

# Finally import scripts
from .src import *
from .linalg import *
from .lax import *
from .autograd import *
from . import scipy
from . import random


