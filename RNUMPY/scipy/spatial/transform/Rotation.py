# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Oct 2024 M. Clarke
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
sp  = rp.scipy_handle
jnp = j.numpy

class Rotation(): 

    @staticmethod
    def apply(R, vectors, inverse=False): 
        return R.apply(vectors, inverse=inverse)
        
    @staticmethod
    def as_euler(R, seq, degrees=False): 
        return R.as_euler(seq=seq, degrees=degrees)
        
    @staticmethod
    def as_matrix(R): 
        return R.as_matrix()
        
    @staticmethod
    def as_mrp(R): 
        return R.as_mrp()
        
    @staticmethod
    def as_quat(R, canonical=False, scalar_first=False): 
        return R.as_quat(canonical=canonical, scalar_first=scalar_first)
        
    @staticmethod
    def as_rotvec(R, degrees=False): 
        return R.as_rotvec(degrees=degrees)
        
    @staticmethod
    def concatenate(rotations): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.concatenate(rotations=rotations)
        else: return j.scipy.spatial.transform.Rotation.concatenate(rotations=rotations)
         
    @staticmethod
    def from_euler(seq, angles, degrees=False): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.from_euler(seq=seq, angles=angles, degrees=degrees)
        else: return j.scipy.spatial.transform.Rotation.from_euler(seq=seq, angles=angles, degrees=degrees)
        
    @staticmethod
    def from_matrix(matrix): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.from_matrix(matrix=matrix)
        else: return j.scipy.spatial.transform.Rotation.from_matrix(matrix=matrix)
        
    @staticmethod
    def from_mrp(mrp): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.from_mrp(mrp=mrp)
        else: return j.scipy.spatial.transform.Rotation.from_mrp(mrp=mrp)
        
    @staticmethod
    def from_quat(quat): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.from_quat(quat=quat)
        else: return j.scipy.spatial.transform.Rotation.from_quat(quat=quat)
        
    @staticmethod
    def from_rotvec(rotvec, degrees=False): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.from_rotvec(rotvec=rotvec, degrees=degrees)
        else: return j.scipy.spatial.transform.Rotation.from_rotvec(rotvec=rotvec, degrees=degrees)
        
    @staticmethod
    def identity(num=None): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.identity(num=num)
        else: return j.scipy.spatial.transform.Rotation.identity(num=num) 
        
    @staticmethod
    def inv(R): 
        return R.inv()
        
    @staticmethod
    def magnitude(R): 
        return R.magnitude()
        
    @staticmethod
    def mean(R, weights=None): 
        return R.mean(weights=weights)
        
    @staticmethod
    def random(num=None, random_state=None): 
        if not rp.use_jax: return sp.spatial.transform.Rotation.random(num=num, random_state=random_state)
        else: return j.scipy.spatial.transform.Rotation.random(num=num, random_state=random_state)     
        
    def count():    raise NotImplementedError 
    def index():    raise NotImplementedError 