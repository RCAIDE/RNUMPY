# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Oct 2024 M. Clarke
# Modified: Apr 2026, E. Botero

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
sp  = rp.scipy_handle
tr  = rp.torch_handle
jnp = j.numpy if j else None
ji  = j.scipy.integrate if j else None
si  = sp.integrate if sp is not None else None
# ti  = tr.integrate if tr else None

def trapezoid(y, x=None, dx=1.0, axis=-1): 
    if rp.use_jax: return jnp.trapezoid(y=y, x=x, dx=dx, axis=axis)
    elif rp.use_torch:
        y_t = tr.as_tensor(y)
        if x is not None:
            x_t = tr.as_tensor(x)
            return rp.TorchArray(tr.trapezoid(y_t, x=x_t, dim=axis))
        else:
            return rp.TorchArray(tr.trapezoid(y_t, dx=dx, dim=axis))
    else: return np.trapezoid(y=y, x=x, dx=dx, axis=axis)

def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    if rp.use_jax:
        import jax.scipy.integrate as jsi
        return jsi.cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)
    elif rp.use_torch:
        y_t = tr.as_tensor(y)
        if x is not None:
            x_t = tr.as_tensor(x)
            res = tr.cumulative_trapezoid(y_t, x=x_t, dim=axis)
        else:
            res = tr.cumulative_trapezoid(y_t, dx=dx, dim=axis)
            
        if initial is not None:
            # Need to prepend the initial value along the given axis
            init_shape = list(res.shape)
            init_shape[axis] = 1
            init_tensor = tr.full(init_shape, initial, dtype=res.dtype, device=res.device)
            res = tr.cat([init_tensor, res], dim=axis)
            
        return rp.TorchArray(res)
    else:
        import scipy.integrate as spi
        return rp.NumpyArray(spi.cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial))