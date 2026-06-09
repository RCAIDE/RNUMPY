# autograd.py
# (c) Copyright 2024 Aerospace Research Community LLC

import RNUMPY as rp

def _to_array(x):
    if rp.use_jax:
        import jax.numpy as jnp
        if not isinstance(x, (rp.JaxArray, jnp.ndarray)):
            return rp.array(x)
    elif rp.use_torch:
        import torch
        if not isinstance(x, (rp.TorchArray, torch.Tensor)):
            return rp.array(x)
    else:
        if not isinstance(x, (rp.NumpyArray, rp.numpy_handle.ndarray)):
            return rp.array(x)
    return x

def _ensure_backend_args(args):
    return tuple(_to_array(arg) for arg in args)

def _finite_diff_grad(f, argnums, args, kwargs, eps=1e-6):
    # args is a tuple of all arguments to f
    # argnums is int or tuple of ints
    
    if isinstance(argnums, int):
        target_indices = [argnums]
        single_arg = True
    else:
        target_indices = argnums
        single_arg = False

    grads = []
    for arg_idx in target_indices:
        x = _to_array(args[arg_idx])
        g = rp.zeros_like(x)
        
        # Flatten for iteration
        x_flat = x.ravel()
        g_flat = g.ravel()
        
        for i in range(x_flat.size):
            orig_val = x_flat[i].item()
            
            # Forward
            x_flat[i] = orig_val + eps
            args_plus = list(args)
            args_plus[arg_idx] = x.reshape(x.shape)
            y_plus = f(*args_plus, **kwargs)
            
            # Backward
            x_flat[i] = orig_val - eps
            args_minus = list(args)
            args_minus[arg_idx] = x.reshape(x.shape)
            y_minus = f(*args_minus, **kwargs)
            
            # Central difference
            g_flat[i] = (y_plus - y_minus) / (2 * eps)
            
            # Reset
            x_flat[i] = orig_val
            
        grads.append(g)
    
    if single_arg:
        return grads[0]
    return tuple(grads)

def _finite_diff_jac(f, argnums, args, kwargs, eps=1e-6):
    if isinstance(argnums, int):
        target_indices = [argnums]
        single_arg = True
    else:
        target_indices = argnums
        single_arg = False

    jacs = []
    for arg_idx in target_indices:
        x = _to_array(args[arg_idx])
        y0 = f(*args, **kwargs)
        y0_flat = _to_array(y0).ravel()
        
        jac = rp.zeros((y0_flat.size, x.size))
        
        x_flat = x.ravel()
        for i in range(x_flat.size):
            orig_val = x_flat[i].item()
            
            x_flat[i] = orig_val + eps
            args_plus = list(args)
            args_plus[arg_idx] = x.reshape(x.shape)
            y_plus = _to_array(f(*args_plus, **kwargs)).ravel()
            
            x_flat[i] = orig_val - eps
            args_minus = list(args)
            args_minus[arg_idx] = x.reshape(x.shape)
            y_minus = _to_array(f(*args_minus, **kwargs)).ravel()
            
            jac[:, i] = (y_plus - y_minus) / (2 * eps)
            
            x_flat[i] = orig_val
            
        # Reshape jacobian to (y_shape, x_shape)
        y_shape = _to_array(y0).shape
        x_shape = x.shape
        full_jac_shape = y_shape + x_shape
        jacs.append(jac.reshape(full_jac_shape))

    if single_arg:
        return jacs[0]
    return tuple(jacs)


def grad(f, argnums=0, has_aux=False):
    def grad_wrapper(*args, **kwargs):
        args = _ensure_backend_args(args)
        if rp.use_jax:
            import jax
            return jax.grad(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
        elif rp.use_torch:
            try:
                from torch.func import grad as tgrad
                return tgrad(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
            except ImportError:
                # Fallback to autograd.grad if torch.func is missing (older torch)
                # This fallback is limited and might not support argnums/has_aux as well as torch.func
                raise ImportError("torch.func is required for RNUMPY.grad in Torch mode. Please upgrade PyTorch.")
        else:
            if has_aux:
                raise NotImplementedError("has_aux=True is not supported in NumPy finite difference mode.")
            return _finite_diff_grad(f, argnums, args, kwargs)
    return grad_wrapper

def value_and_grad(f, argnums=0, has_aux=False):
    def v_and_g_wrapper(*args, **kwargs):
        args = _ensure_backend_args(args)
        if rp.use_jax:
            import jax
            return jax.value_and_grad(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
        elif rp.use_torch:
            try:
                from torch.func import grad_and_value as tgrad_and_value
                g, v = tgrad_and_value(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
                return v, g
            except ImportError:
                raise ImportError("torch.func.grad_and_value is required for RNUMPY.value_and_grad in Torch mode.")
        else:
            if has_aux:
                raise NotImplementedError("has_aux=True is not supported in NumPy finite difference mode.")
            val = f(*args, **kwargs)
            g = _finite_diff_grad(f, argnums, args, kwargs)
            return val, g
    return v_and_g_wrapper

def jacfwd(f, argnums=0, has_aux=False):
    def jacfwd_wrapper(*args, **kwargs):
        args = _ensure_backend_args(args)
        if rp.use_jax:
            import jax
            return jax.jacfwd(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
        elif rp.use_torch:
            try:
                from torch.func import jacfwd as tjacfwd
                return tjacfwd(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
            except ImportError:
                raise ImportError("torch.func is required for RNUMPY.jacfwd in Torch mode.")
        else:
            if has_aux:
                raise NotImplementedError("has_aux=True is not supported in NumPy finite difference mode.")
            return _finite_diff_jac(f, argnums, args, kwargs)
    return jacfwd_wrapper

def jacrev(f, argnums=0, has_aux=False):
    def jacrev_wrapper(*args, **kwargs):
        args = _ensure_backend_args(args)
        if rp.use_jax:
            import jax
            return jax.jacrev(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
        elif rp.use_torch:
            try:
                from torch.func import jacrev as tjacrev
                return tjacrev(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
            except ImportError:
                raise ImportError("torch.func is required for RNUMPY.jacrev in Torch mode.")
        else:
            if has_aux:
                raise NotImplementedError("has_aux=True is not supported in NumPy finite difference mode.")
            return _finite_diff_jac(f, argnums, args, kwargs)
    return jacrev_wrapper

def jacobian(f, argnums=0, has_aux=False):
    # Default to jacrev as it's often more efficient for many outputs
    return jacrev(f, argnums=argnums, has_aux=has_aux)

def hessian(f, argnums=0, has_aux=False):
    def hessian_wrapper(*args, **kwargs):
        args = _ensure_backend_args(args)
        if rp.use_jax:
            import jax
            return jax.hessian(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
        elif rp.use_torch:
            try:
                from torch.func import hessian as thessian
                # torch.func.hessian does not accept has_aux
                return thessian(f, argnums=argnums)(*args, **kwargs)
            except ImportError:
                raise ImportError("torch.func is required for RNUMPY.hessian in Torch mode.")
        else:
            raise NotImplementedError("Hessian is not implemented for NumPy mode yet.")
    return hessian_wrapper

def jvp(f, primals, tangents):
    primals = _ensure_backend_args(primals)
    tangents = _ensure_backend_args(tangents)
    if rp.use_jax:
        import jax
        return jax.jvp(f, primals, tangents)
    elif rp.use_torch:
        try:
            from torch.func import jvp as tjvp
            return tjvp(f, primals, tangents)
        except ImportError:
            raise ImportError("torch.func is required for RNUMPY.jvp in Torch mode.")
    else:
        # jvp(f, (x,), (v,)) = grad(f)(x) @ v
        # For simplicity, implement via finite difference: (f(x + eps*v) - f(x - eps*v)) / (2*eps)
        eps = 1e-6
        
        args_plus = []
        args_minus = []
        for p, t in zip(primals, tangents):
            args_plus.append(p + eps * t)
            args_minus.append(p - eps * t)
            
        y_plus = f(*args_plus)
        y_minus = f(*args_minus)
        
        y0 = f(*primals)
        return y0, (y_plus - y_minus) / (2 * eps)

def vjp(f, *primals, has_aux=False):
    primals = _ensure_backend_args(primals)
    if rp.use_jax:
        import jax
        return jax.vjp(f, *primals, has_aux=has_aux)
    elif rp.use_torch:
        try:
            from torch.func import vjp as tvjp
            return tvjp(f, *primals, has_aux=has_aux)
        except ImportError:
            raise ImportError("torch.func is required for RNUMPY.vjp in Torch mode.")
    else:
        # vjp is harder to implement generically via finite differences without full jacobian
        # but JAX's vjp returns (y, vjp_fun)
        y0 = f(*primals)
        def vjp_fun(v):
            # grad(v^T f(x))
            def scalar_fun(*args):
                return rp.dot(v, f(*args))
            return grad(scalar_fun)(*primals),
            
        if has_aux:
             # JAX vjp behavior with has_aux returning (y, aux, vjp_fun)
             if isinstance(y0, tuple) and len(y0) == 2:
                  return y0[0], y0[1], vjp_fun
             return y0, None, vjp_fun # Fallback
        return y0, vjp_fun
