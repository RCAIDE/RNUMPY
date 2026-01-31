# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Oct 2024 M. Clarke
# Modified: Jan 2026, E. Botero

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp
from RNUMPY import NumpyArray
import scipy.spatial.transform

if rp.jax_handle is not None:
    try:
        import jax.scipy.spatial.transform
    except ImportError:
        pass

j   = rp.jax_handle
np  = rp.numpy_handle
sp  = rp.scipy_handle
jnp = j.numpy if j is not None else None
jspatial = j.scipy.spatial.transform if j is not None else None

class Rotation(): 
    def __init__(self, native):
        self._native = native

    # --- Methods supporting both Instance and Functional API ---

    def apply(R, vectors, inverse=False): 
        native_R = getattr(R, '_native', R)
        res = native_R.apply(vectors, inverse=inverse)
        if not rp.use_jax: return rp.NumpyArray(res)
        else: return res
        
    def as_euler(R, seq, degrees=False): 
        native_R = getattr(R, '_native', R)
        res = native_R.as_euler(seq=seq, degrees=degrees)
        if not rp.use_jax: return rp.NumpyArray(res)
        else: return res
        
    def as_matrix(R): 
        native_R = getattr(R, '_native', R)
        res = native_R.as_matrix()
        if not rp.use_jax: return rp.NumpyArray(res)
        else: return res
        
    def as_mrp(R): 
        native_R = getattr(R, '_native', R)
        res = native_R.as_mrp()
        if not rp.use_jax: return rp.NumpyArray(res)
        else: return res
        
    def as_quat(R, canonical=False, scalar_first=False): 
        native_R = getattr(R, '_native', R)
        res = native_R.as_quat(canonical=canonical, scalar_first=scalar_first)
        if not rp.use_jax: return rp.NumpyArray(res)
        else: return res
        
    def as_rotvec(R, degrees=False): 
        native_R = getattr(R, '_native', R)
        res = native_R.as_rotvec(degrees=degrees)
        if not rp.use_jax: return rp.NumpyArray(res)
        else: return res

    def inv(R): 
        native_R = getattr(R, '_native', R)
        res = native_R.inv()
        if not rp.use_jax: return Rotation(res)
        else: return res
        
    def magnitude(R): 
        native_R = getattr(R, '_native', R)
        res = native_R.magnitude()
        if not rp.use_jax: return rp.NumpyArray(res)
        else: return res
        
    def mean(R, weights=None): 
        native_R = getattr(R, '_native', R)
        res = native_R.mean(weights=weights)
        if not rp.use_jax: return Rotation(res)
        else: return res

    def __mul__(self, other):
        native_self = getattr(self, '_native', self)
        other_native = getattr(other, '_native', other)
        res = native_self * other_native
        if not rp.use_jax: return Rotation(res)
        else: return res

    def __getitem__(self, key):
        res = self._native[key]
        if not rp.use_jax: return Rotation(res)
        else: return res

    def __len__(self):
        return len(self._native)

    def __repr__(self):
        return f"RNUMPY.Rotation({repr(self._native)})"

    # --- Static Factory Methods ---

    @staticmethod
    def concatenate(rotations): 
        native_rotations = [getattr(r, '_native', r) for r in rotations]
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.concatenate(rotations=native_rotations)
            return Rotation(res)
        else: 
            return jspatial.Rotation.concatenate(rotations=native_rotations)
         
    @staticmethod
    def from_euler(seq, angles, degrees=False): 
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.from_euler(seq=seq, angles=angles, degrees=degrees)
            return Rotation(res)
        else: 
            return jspatial.Rotation.from_euler(seq=seq, angles=angles, degrees=degrees)
        
    @staticmethod
    def from_matrix(matrix): 
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.from_matrix(matrix=matrix)
            return Rotation(res)
        else: 
            return jspatial.Rotation.from_matrix(matrix=matrix)
        
    @staticmethod
    def from_mrp(mrp): 
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.from_mrp(mrp=mrp)
            return Rotation(res)
        else: 
            return jspatial.Rotation.from_mrp(mrp=mrp)
        
    @staticmethod
    def from_quat(quat): 
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.from_quat(quat=quat)
            return Rotation(res)
        else: 
            return jspatial.Rotation.from_quat(quat=quat)
        
    @staticmethod
    def from_rotvec(rotvec, degrees=False): 
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.from_rotvec(rotvec=rotvec, degrees=degrees)
            return Rotation(res)
        else: 
            return jspatial.Rotation.from_rotvec(rotvec=rotvec, degrees=degrees)
        
    @staticmethod
    def identity(num=None): 
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.identity(num=num)
            return Rotation(res)
        else: 
            return jspatial.Rotation.identity(num=num) 
        
    @staticmethod
    def random(num=None, random_state=None): 
        if not rp.use_jax: 
            res = sp.spatial.transform.Rotation.random(num=num, random_state=random_state)
            return Rotation(res)
        else: 
            return jspatial.Rotation.random(num=num, random_state=random_state)     
        
    def count():    raise NotImplementedError 
    def index():    raise NotImplementedError