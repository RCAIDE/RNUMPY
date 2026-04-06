# optimize.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Oct 2024 M. Clarke
# Modified: March 2026 E. Botero (Add fmin_slsqp with gradients)

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
sp  = rp.scipy_handle
tr  = rp.torch_handle

if j is not None:
    try:
        import jax.scipy.optimize
    except ImportError:
        pass

jo  = j.scipy.optimize if j else None
so  = sp.optimize if sp is not None else None

# ----------------------------------------------------------------------------------------------------------------------
#  Functions
# ----------------------------------------------------------------------------------------------------------------------  

def minimize(fun, x0, args=(), *, method='BFGS', bounds=None, constraints=(), tol=None, options=None): 
    if rp.use_jax: 
        if bounds is not None:
             raise NotImplementedError('bounds are not supported for minimize in JAX')
        if constraints and len(constraints) > 0:
             raise NotImplementedError('constraints are not supported for minimize in JAX')
        return jo.minimize(fun, x0, args=args, method=method, tol=tol, options=options)
    elif rp.use_torch: raise NotImplementedError('minimize not supported for Torch here')
    else: return so.minimize(fun, x0, args=args, method=method, bounds=bounds, constraints=constraints, tol=tol, options=options)

def fmin_slsqp(func, x0, fprime=None, f_eqcons=None, fprime_eqcons=None, 
               f_ieqcons=None, fprime_ieqcons=None, bounds=(), iter=100, 
               acc=1e-06, iprint=1, disp=None, full_output=0, 
               epsilon=1.4901161193847656e-08, callback=None, args=()): 
    
    

    if rp.use_jax:
        import jax
        jnp = jax.numpy

        def wrap_func(x_np, *args_passthrough):
            x_jax = jnp.array(x_np)
            res = func(x_jax, *args_passthrough)
            return np.array(res)

        def wrap_fprime(x_np, *args_passthrough):
            x_jax = jnp.array(x_np)
            grad_func = jax.grad(lambda x: func(x, *args_passthrough).sum())
            return np.array(grad_func(x_jax))

        fprime_to_use = wrap_fprime if fprime is None else fprime

        # Handle constraints
        wrapped_f_eqcons = None
        wrapped_fprime_eqcons = None
        if f_eqcons is not None:
             def wrapped_f_eqcons(x_np, *args_passthrough):
                  return np.array(f_eqcons(jnp.array(x_np), *args_passthrough))
             if fprime_eqcons is None:
                  def wrapped_fprime_eqcons(x_np, *args_passthrough):
                       jac_func = jax.jacobian(f_eqcons)
                       return np.array(jac_func(jnp.array(x_np), *args_passthrough))
             else:
                  wrapped_fprime_eqcons = fprime_eqcons

        wrapped_f_ieqcons = None
        wrapped_fprime_ieqcons = None
        if f_ieqcons is not None:
             def wrapped_f_ieqcons(x_np, *args_passthrough):
                  return np.array(f_ieqcons(jnp.array(x_np), *args_passthrough))
             if fprime_ieqcons is None:
                  def wrapped_fprime_ieqcons(x_np, *args_passthrough):
                       jac_func = jax.jacobian(f_ieqcons)
                       return np.array(jac_func(jnp.array(x_np), *args_passthrough))
             else:
                  wrapped_fprime_ieqcons = fprime_ieqcons

        x0_np = np.array(x0)
        res = so.fmin_slsqp(wrap_func, x0_np, fprime=fprime_to_use, 
                            f_eqcons=wrapped_f_eqcons, fprime_eqcons=wrapped_fprime_eqcons, 
                            f_ieqcons=wrapped_f_ieqcons, fprime_ieqcons=wrapped_fprime_ieqcons, 
                            bounds=bounds, iter=iter, acc=acc, iprint=iprint, disp=disp, 
                            full_output=full_output, epsilon=epsilon, callback=callback, args=args)
        
        if full_output:
             x, obj, niter, imode, smessage = res
             return rp.array(x), obj, niter, imode, smessage
        else:
             return rp.array(res)

    elif rp.use_torch:
        import torch as tr

        dt = x0.dtype
        
        def wrap_func(x_np, *args_passthrough):
            x_tr = rp.array(x_np, dtype=dt)
            res = func(x_tr, *args_passthrough)
            return tr.as_tensor(res).detach().cpu().numpy()

        def wrap_fprime(x_np, *args_passthrough):
            x_tr = tr.tensor(x_np, dtype=dt, requires_grad=True)
            res = func(rp.TorchArray(x_tr), *args_passthrough)
            res_sum = tr.as_tensor(res).sum()
            grad = tr.autograd.grad(res_sum, x_tr)[0]
            return grad.detach().cpu().numpy()

        fprime_to_use = wrap_fprime if fprime is None else fprime

        # Handle constraints
        wrapped_f_eqcons = None
        wrapped_fprime_eqcons = None
        if f_eqcons is not None:
             def wrapped_f_eqcons(x_np, *args_passthrough):
                  x_tr = tr.as_tensor(x_np, dtype=dt)
                  return tr.as_tensor(f_eqcons(rp.TorchArray(x_tr), *args_passthrough)).detach().cpu().numpy()
             if fprime_eqcons is None:
                  def wrapped_fprime_eqcons(x_np, *args_passthrough):
                       x_tr = tr.as_tensor(x_np, dtype=dt)
                       # Check for empty constraints
                       test_out = tr.as_tensor(f_eqcons(rp.TorchArray(x_tr), *args_passthrough))
                       if test_out.nelement() == 0:
                            return np.zeros((0, len(x_np)))
                       
                       def func_for_jac(x_t):
                            return tr.as_tensor(f_eqcons(rp.TorchArray(x_t), *args_passthrough))
                       jac = tr.autograd.functional.jacobian(func_for_jac, x_tr)
                       return jac.detach().cpu().numpy()
             else:
                  wrapped_fprime_eqcons = fprime_eqcons

        wrapped_f_ieqcons = None
        wrapped_fprime_ieqcons = None
        if f_ieqcons is not None:
             def wrapped_f_ieqcons(x_np, *args_passthrough):
                  x_tr = tr.as_tensor(x_np, dtype=dt)
                  return tr.as_tensor(f_ieqcons(rp.TorchArray(x_tr), *args_passthrough)).detach().cpu().numpy()
             if fprime_ieqcons is None:
                  def wrapped_fprime_ieqcons(x_np, *args_passthrough):
                       x_tr = tr.as_tensor(x_np, dtype=dt)
                       # Check for empty constraints
                       test_out = tr.as_tensor(f_ieqcons(rp.TorchArray(x_tr), *args_passthrough))
                       if test_out.nelement() == 0:
                            return np.zeros((0, len(x_np)))

                       def func_for_jac(x_t):
                            return tr.as_tensor(f_ieqcons(rp.TorchArray(x_t), *args_passthrough))
                       jac = tr.autograd.functional.jacobian(func_for_jac, x_tr)
                       return jac.detach().cpu().numpy()
             else:
                  wrapped_fprime_ieqcons = fprime_ieqcons

        x0_np = np.asarray(tr.as_tensor(x0).detach().cpu())
        res = so.fmin_slsqp(wrap_func, x0_np, fprime=fprime_to_use, 
                            f_eqcons=wrapped_f_eqcons, fprime_eqcons=wrapped_fprime_eqcons, 
                            f_ieqcons=wrapped_f_ieqcons, fprime_ieqcons=wrapped_fprime_ieqcons, 
                            bounds=bounds, iter=iter, acc=acc, iprint=iprint, disp=disp, 
                            full_output=full_output, epsilon=epsilon, callback=callback, args=args)
        
        if full_output:
             x, obj, niter, imode, smessage = res
             return rp.array(x), obj, niter, imode, smessage
        else:
             return rp.array(res)

    else:
        return so.fmin_slsqp(func, x0, fprime=fprime, f_eqcons=f_eqcons, fprime_eqcons=fprime_eqcons, 
                             f_ieqcons=f_ieqcons, fprime_ieqcons=fprime_ieqcons, bounds=bounds, iter=iter, 
                             acc=acc, iprint=iprint, disp=disp, full_output=full_output, 
                             epsilon=epsilon, callback=callback, args=args)

def OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit): 
    if rp.use_jax: return jo.OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit)
    elif rp.use_torch: raise NotImplementedError('OptimizeResults not supported for Torch')
    else: return so.OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit)