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
tr  = rp.torch_handle
jnp = j.numpy if j is not None else None
if j is not None:
    try:
        import jax.scipy.optimize
    except ImportError:
        pass

joptmin = j.scipy.optimize.minimize if j is not None else None

def fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None):
    if rp.use_jax:
        return _jax_fsolve(func,x0,args,maxfev,xtol)        
    elif rp.use_torch:
        return _pytorch_fsolve(func, x0, args, fprime, full_output, col_deriv, xtol, maxfev, band, epsfcn, factor, diag)
    else:
        return sp.optimize.fsolve(func,x0,args,fprime,full_output,col_deriv,xtol,maxfev,band,epsfcn,factor,diag)

def _pytorch_fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None):
    # This implementation uses the Implicit Function Theorem (IFT) to provide gradients
    # for the solution x w.r.t the parameters p that func depends on.
    # IFT: df/dx * dx/dp + df/dp = 0  => dx/dp = - (df/dx)^-1 * df/dp
    
    dt = x0.dtype
    device = x0.device if hasattr(x0, 'device') else None

    # Helper function to find and replace tensors in nested structures
    def find_tensors(obj):
        tensors = []
        if isinstance(obj, tr.Tensor):
            tensors.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                tensors.extend(find_tensors(item))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                tensors.extend(find_tensors(v))
        elif hasattr(obj, 'items'): # Handle RCAIDE.Data objects
            for k, v in obj.items():
                tensors.extend(find_tensors(v))
        return tensors

    def replace_tensors(obj, tensors, state):
        if isinstance(obj, tr.Tensor):
            val = tensors[state['idx']]
            state['idx'] += 1
            return val
        elif type(obj) is list:
            return [replace_tensors(item, tensors, state) for item in obj]
        elif type(obj) is tuple:
            return tuple(replace_tensors(item, tensors, state) for item in obj)
        elif type(obj) is dict:
            return {k: replace_tensors(v, tensors, state) for k, v in obj.items()}
        elif hasattr(obj, 'items') and hasattr(obj, 'copy'): # Handle custom dict-like objects
            import copy
            new_obj = copy.copy(obj)
            for k, v in obj.items():
                new_obj[k] = replace_tensors(v, tensors, state)
            return new_obj
        return obj

    class FSolve(tr.autograd.Function):
        @staticmethod
        def forward(ctx, x0_t, *params):
            # params are the tensors extracted from args
            
            def func_np(x_np):
                x_tr = tr.as_tensor(x_np, dtype=dt, device=device)
                
                # Reconstruct args for the function call
                current_args = replace_tensors(args, params, {'idx': 0})
                
                res = func(rp.TorchArray(x_tr), *current_args)
                return tr.as_tensor(res).detach().cpu().numpy()

            x_sol_np = sp.optimize.fsolve(func_np, x0_t.detach().cpu().numpy(), 
                                           xtol=xtol, maxfev=maxfev, band=band, 
                                           epsfcn=epsfcn, factor=factor, diag=diag)
            
            x_sol = tr.as_tensor(x_sol_np, dtype=dt, device=device)
            ctx.save_for_backward(x_sol, *params)
            return x_sol

        @staticmethod
        def backward(ctx, grad_x):
            x_sol = ctx.saved_tensors[0]
            params = ctx.saved_tensors[1:]
            
            # We need to compute df/dx and df/dp at x_sol
            with tr.enable_grad():
                x = x_sol.detach().requires_grad_(True)
                
                # Reconstruct args with potentially tracked parameters
                current_args = replace_tensors(args, params, {'idx': 0})
                
                f = func(rp.TorchArray(x), *current_args)
                f_tensor = tr.as_tensor(f)

            # 1. Compute df/dx (Jacobian)
            def f_wrapper(x_val):
                return tr.as_tensor(func(rp.TorchArray(x_val), *current_args))
            
            df_dx = tr.autograd.functional.jacobian(f_wrapper, x)
            
            # 2. Vector-Jacobian Product (VJP) for parameters
            try:
                lambda_ = tr.linalg.solve(df_dx.T, grad_x.reshape(-1, 1)).reshape(-1)
            except RuntimeError as e:
                print(f"WARNING: tr.linalg.solve failed: {e}")
                lambda_ = tr.linalg.lstsq(df_dx.T, grad_x.reshape(-1, 1)).solution.reshape(-1)

            # Diagnostic prints
            if tr.any(tr.isnan(f_tensor)): print("DEBUG backward: f_tensor contains NaNs")
            if tr.any(tr.isnan(df_dx)): print("DEBUG backward: df_dx contains NaNs")
            if tr.any(tr.isnan(lambda_)): print("DEBUG backward: lambda_ contains NaNs")
                
            # Filter params that require grad
            grad_params = [None] * len(params)
            params_to_diff = []
            params_indices = []
            for i, p in enumerate(params):
                if p.requires_grad:
                    params_to_diff.append(p)
                    params_indices.append(i)
            
            if params_to_diff:
                grads_p = tr.autograd.grad(f_tensor, params_to_diff, grad_outputs=-lambda_, retain_graph=True, allow_unused=True)
                for i, g in zip(params_indices, grads_p):
                    grad_params[i] = g
            
            return (None, *grad_params)

    # To use FSolve, we need to pass the parameters explicitly.
    params = find_tensors(args)
    
    x0_tr = tr.as_tensor(x0, dtype=dt, device=device)
    res_x = FSolve.apply(x0_tr, *params)
    
    if full_output:
        # For full_output, we run fsolve once more without autograd to get other outputs.
        x_np, infodict, ier, mesg = sp.optimize.fsolve(lambda x: func(rp.TorchArray(tr.as_tensor(x, dtype=dt, device=device)), *args).detach().cpu().numpy(), 
                                                        x0_tr.detach().cpu().numpy(), 
                                                        args=tuple(), 
                                                        xtol=xtol, maxfev=maxfev, band=band, 
                                                        epsfcn=epsfcn, factor=factor, diag=diag,
                                                        full_output=True)
        return rp.array(res_x, dtype=dt), infodict, ier, mesg
    else:
        return rp.array(res_x, dtype=dt)

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