# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Aug 2024 E. Botero
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
sp  = rp.scipy_handle
jnp = j.numpy
if rp.jax_handle is not None:
    try:
        import jax.scipy.optimize
    except ImportError:
        pass

joptmin = j.scipy.optimize.minimize if j is not None else None

def fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None):
    if rp.use_jax:
        return _jax_fsolve(func,x0,args,maxfev,xtol)        
    else:
        return sp.optimize.fsolve(func,x0,args,fprime,full_output,col_deriv,xtol,maxfev,band,epsfcn,factor,diag)

def _jax_fsolve(func,x0,args,maxfev,tol):

    # a wrapper to make it into least squares form
    def wrap(x):
        return 0.5*jnp.sum(func(x)**2)

    # coax the inputs to the correct format
    options={'maxiter':maxfev}

    # run jax minimize on BFGS
    # TODO: make the tol's consistent with original scipy version
    OR = joptmin(wrap,x0,args,method='BFGS',tol=tol,options=options)

    # Unpack into the same format as scipy
    x        = OR.x
    infodict = {'nfev':OR.nfev,'njev':OR.njev,'fvec':OR.fun,'fjac':OR.jac,'r':None,'qtf':None}
    ier      = OR.success
    mesg     = OR.status

    return x, infodict, ier, mesg 