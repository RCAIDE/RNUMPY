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
tr  = rp.torch_handle

if j is not None:
    try:
        import jax.scipy.interpolate
    except ImportError:
        pass

ji  = j.scipy.interpolate if j else None
si  = sp.interpolate if sp is not None else None
TorchArray = rp.TorchArray


def RegularGridInterpolator(points, values, method='linear', bounds_error=False, fill_value=np.nan): 
    if rp.use_jax: 
        f = ji.RegularGridInterpolator(points=points, values=values, method=method, bounds_error=bounds_error, fill_value=fill_value)
        def wrapped_ji_interp(xi):
             return rp.array(f(xi))
        return wrapped_ji_interp
    elif rp.use_torch:
        return _TorchRegularGridInterpolator(points=points, values=values, method=method, bounds_error=bounds_error, fill_value=fill_value)
    else:
        f = si.RegularGridInterpolator(points=points, values=values, method=method, bounds_error=bounds_error, fill_value=fill_value)
        def wrapped_si_interp(xi):
             return rp.array(f(xi))
        return wrapped_si_interp

def interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=False):
    if rp.use_jax:
        jnp = rp.jax_handle.numpy
        if kind == 'linear':
             def wrapped_jnp_interp_linear(x_new):
                  return jnp.interp(x_new, x, y)
             return wrapped_jnp_interp_linear
        elif kind == 'cubic':
             x_j = jnp.asarray(x)
             y_j = jnp.asarray(y)
             tck = splrep(x_j, y_j, k=3, s=0)
             def jax_interp_cubic(x_new):
                 x_new_j = jnp.asarray(x_new)
                 x_new_j = jnp.clip(x_new_j, x_j[0], x_j[-1])
                 return rp.array(splev(x_new_j, tck))
             return jax_interp_cubic
        raise NotImplementedError(f'JAX interp1d kind {kind} not implemented')
        
    elif rp.use_torch:
        if kind == 'linear':
            x_t = tr.as_tensor(x)
            y_t = tr.as_tensor(y)
            def torch_interp_linear(x_new):
                x_new_t = tr.as_tensor(x_new)
                indices = tr.searchsorted(x_t, x_new_t)
                indices = tr.clamp(indices, 1, len(x_t) - 1)
                x_left = x_t[indices - 1]
                x_right = x_t[indices]
                y_left = y_t[indices - 1]
                y_right = y_t[indices]
                weight = (x_new_t - x_left) / (x_right - x_left)
                res = y_left + weight * (y_right - y_left)
                return TorchArray(res)
            return torch_interp_linear
        elif kind == 'cubic':
            x_t = tr.as_tensor(x)
            y_t = tr.as_tensor(y)
            tck = splrep(x_t, y_t, k=3, s=0)
            def torch_interp_cubic(x_new):
                x_new_t = tr.as_tensor(x_new)
                x_new_t = tr.clamp(x_new_t, x_t[0], x_t[-1])
                return TorchArray(splev(x_new_t, tck))
            return torch_interp_cubic
        raise NotImplementedError(f'PyTorch interp1d kind {kind} not implemented')
    
    else:
        f = si.interp1d(rp.array(x), rp.array(y), kind=kind, axis=axis, copy=copy, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
        def wrapped_si_interp(x_new):
             return rp.array(f(rp.array(x_new)))
        return wrapped_si_interp

def _torch_bspline_basis(x_eval, t, k):
    n_basis = len(t) - k - 1
    x_eval = x_eval.unsqueeze(0)
    t_view = t.unsqueeze(1)
    
    B = ((t_view[:-1] <= x_eval) & (x_eval < t_view[1:])).to(x_eval.dtype)
    
    for d in range(1, k + 1):
        # Basis of degree d depends on knots t[i], t[i+d], t[i+1], t[i+d+1]
        # and basis of degree d-1
        t_i = t[:-d-1]
        t_id = t[d:-1]
        t_i1 = t[1:-d]
        t_id1 = t[d+1:]
        
        denom1 = t_id - t_i
        mask1 = denom1 > 0
        denom1_safe = tr.where(mask1, denom1, tr.ones_like(denom1))
        T1 = ((x_eval - t_i.unsqueeze(1)) / denom1_safe.unsqueeze(1)) * B[:-1]
        term1 = tr.where(mask1.unsqueeze(1), T1, tr.zeros_like(T1))
        
        denom2 = t_id1 - t_i1
        mask2 = denom2 > 0
        denom2_safe = tr.where(mask2, denom2, tr.ones_like(denom2))
        T2 = ((t_id1.unsqueeze(1) - x_eval) / denom2_safe.unsqueeze(1)) * B[1:]
        term2 = tr.where(mask2.unsqueeze(1), T2, tr.zeros_like(T2))
        
        B = term1 + term2

    is_right_boundary = (x_eval == t[-1]).squeeze(0)
    B[-1, is_right_boundary] = 1.0
    
    return B.T

def _jax_bspline_basis(x_eval, t, k):
    jnp = rp.jax_handle.numpy
    n_basis = len(t) - k - 1
    x_eval = x_eval[jnp.newaxis, :]
    t_view = t[:, jnp.newaxis]
    
    B = jnp.logical_and(t_view[:-1] <= x_eval, x_eval < t_view[1:]).astype(x_eval.dtype)
    
    for d in range(1, k + 1):
        t_i = t[:-d-1]
        t_id = t[d:-1]
        t_i1 = t[1:-d]
        t_id1 = t[d+1:]
        
        denom1 = t_id - t_i
        denom1_safe = jnp.where(denom1 > 0, denom1, 1.0)
        mask1 = denom1 > 0
        T1 = ((x_eval - t_i[:, jnp.newaxis]) / denom1_safe[:, jnp.newaxis]) * B[:-1]
        term1 = jnp.where(mask1[:, jnp.newaxis], T1, jnp.zeros_like(T1))
        
        denom2 = t_id1 - t_i1
        denom2_safe = jnp.where(denom2 > 0, denom2, 1.0)
        mask2 = denom2 > 0
        T2 = ((t_id1[:, jnp.newaxis] - x_eval) / denom2_safe[:, jnp.newaxis]) * B[1:]
        term2 = jnp.where(mask2[:, jnp.newaxis], T2, jnp.zeros_like(T2))
        
        B = term1 + term2

    is_right_boundary = (x_eval == t[-1])[0]
    B = B.at[-1].set(jnp.where(is_right_boundary, 1.0, B[-1]))
    
    return B.T

def splprep(x, w=None, u=None, ub=None, ue=None, k=3, s=None, per=0, quiet=1):
    if (rp.use_jax or rp.use_torch) and (s == 0 or s is None):
        if rp.use_torch:
            arrays = [tr.as_tensor(arr) for arr in x]
            n = len(arrays[0])
            if u is None:
                diffs = tr.stack([arr[1:] - arr[:-1] for arr in arrays])
                dist = (diffs**2).sum(dim=0).sqrt()
                u_knots = tr.cat([tr.zeros(1, device=dist.device, dtype=dist.dtype), tr.cumsum(dist, dim=0)])
                if u_knots[-1] > 0:
                    u_knots = u_knots / u_knots[-1]
            else:
                u_knots = tr.as_tensor(u)
                
            t_knots = tr.zeros(n + k + 1, dtype=u_knots.dtype, device=u_knots.device)
            t_knots[:k+1] = u_knots[0]
            t_knots[-k-1:] = u_knots[-1]
            if n > 2 * k - 2:
                trim = (k + 1) // 2
                t_knots[k+1:n] = u_knots[trim:-trim]
                
            A = _torch_bspline_basis(u_knots, t_knots, k)
            coeffs = []
            for arr in arrays:
                c_inner = tr.linalg.solve(A, arr)
                c = tr.zeros(len(t_knots), dtype=c_inner.dtype, device=c_inner.device)
                c[:n] = c_inner
                coeffs.append(c)
                
                
            tck = (rp.array(t_knots), [rp.array(c) for c in coeffs], k)
            return tck, rp.array(u_knots)
            
        elif rp.use_jax:
            jnp = rp.jax_handle.numpy
            arrays = [jnp.asarray(arr) for arr in x]
            n = len(arrays[0])
            if u is None:
                diffs = jnp.stack([arr[1:] - arr[:-1] for arr in arrays], axis=0)
                dist = jnp.sqrt((diffs**2).sum(axis=0))
                u_knots = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dist)])
                if u_knots[-1] > 0:
                    u_knots = u_knots / u_knots[-1]
            else:
                u_knots = jnp.asarray(u)
                
            t_knots = jnp.zeros(n + k + 1, dtype=u_knots.dtype)
            t_knots = t_knots.at[:k+1].set(u_knots[0])
            t_knots = t_knots.at[-k-1:].set(u_knots[-1])
            if n > 2 * k - 2:
                trim = (k + 1) // 2
                t_knots = t_knots.at[k+1:n].set(u_knots[trim:-trim])
                
            A = _jax_bspline_basis(u_knots, t_knots, k)
            coeffs = []
            for arr in arrays:
                c_inner = jnp.linalg.solve(A, arr)
                c = jnp.zeros(len(t_knots), dtype=c_inner.dtype)
                c = c.at[:n].set(c_inner)
                coeffs.append(c)
                
                
            tck = (rp.array(t_knots), [rp.array(c) for c in coeffs], k)
            return tck, rp.array(u_knots)

    if rp.use_jax or rp.use_torch:
         raise NotImplementedError("Differentiable splprep only implemented for interpolation (s=0) for now.")
         
    x_np = [np.array(arr) for arr in x]
    tck, u_out = si.splprep(x_np, w=w, u=u, ub=ub, ue=ue, k=k, s=s, per=per, quiet=quiet)
    return (rp.array(tck[0]), [rp.array(c) for c in tck[1]], tck[2]), rp.array(u_out)

def splev(x, tck, der=0, ext=0):
    if (rp.use_torch or rp.use_jax) and isinstance(tck, tuple) and len(tck) == 3 and der == 0:
        t, c, k = tck
        c_list = c if isinstance(c, list) else [c]
        
        if rp.use_torch:
            x_in = tr.as_tensor(x)
            A = _torch_bspline_basis(x_in, t, k)
            n_basis = len(t) - k - 1
            results = []
            for c_arr in c_list:
                c_n = c_arr[:n_basis]
                res = tr.matmul(A, c_n.to(A.dtype))
                results.append(TorchArray(res))
        elif rp.use_jax:
            jnp = rp.jax_handle.numpy
            x_in = jnp.asarray(x)
            A = _jax_bspline_basis(x_in, t, k)
            n_basis = len(t) - k - 1
            results = []
            for c_arr in c_list:
                c_n = c_arr[:n_basis]
                res = jnp.dot(A, c_n.astype(A.dtype))
                results.append(rp.array(res))
                
        if not isinstance(c, list):
            return results[0]
        return results

    if rp.use_jax or rp.use_torch:
         raise NotImplementedError("Differentiable splev only implemented for evaluation (der=0) for now.")

    res = si.splev(x, tck, der=der, ext=ext)
    if isinstance(res, list):
         return [rp.array(r) for r in res]
    return rp.array(res)

def splrep(x, y, w=None, xb=None, xe=None, k=3, s=None, t=None, task=0, full_output=0, per=0, quiet=1):
    if (rp.use_torch or rp.use_jax) and (s == 0 or s is None):
        if rp.use_torch:
            x_t = tr.as_tensor(x)
            y_t = tr.as_tensor(y)
            n = len(x_t)
            
            t_knots = tr.zeros(n + k + 1, dtype=x_t.dtype, device=x_t.device)
            t_knots[:k+1] = x_t[0]
            t_knots[-k-1:] = x_t[-1]
            if n > 2 * k - 2:
                trim = (k + 1) // 2
                t_knots[k+1:n] = x_t[trim:-trim]
                
            A = _torch_bspline_basis(x_t, t_knots, k)
            c_inner = tr.linalg.solve(A, y_t)
            
            c = tr.zeros(len(t_knots), dtype=c_inner.dtype, device=c_inner.device)
            c[:n] = c_inner
            
            tck = (rp.array(t_knots), rp.array(c), k)
            return tck
            
        elif rp.use_jax:
            jnp = rp.jax_handle.numpy
            x_j = jnp.asarray(x)
            y_j = jnp.asarray(y)
            n = len(x_j)
            
            t_knots = jnp.zeros(n + k + 1, dtype=x_j.dtype)
            t_knots = t_knots.at[:k+1].set(x_j[0])
            t_knots = t_knots.at[-k-1:].set(x_j[-1])
            if n > 2 * k - 2:
                trim = (k + 1) // 2
                t_knots = t_knots.at[k+1:n].set(x_j[trim:-trim])
                
            A = _jax_bspline_basis(x_j, t_knots, k)
            c_inner = jnp.linalg.solve(A, y_j)
            
            c = jnp.zeros(len(t_knots), dtype=c_inner.dtype)
            c = c.at[:n].set(c_inner)
            
            tck = (rp.array(t_knots), rp.array(c), k)
            return tck
    
    if rp.use_jax or rp.use_torch:
         raise NotImplementedError("Differentiable splrep only implemented for interpolation (s=0) for now.")

    res = si.splrep(x, y, w=w, xb=xb, xe=xe, k=k, s=s, t=t, task=task, full_output=full_output, per=per, quiet=quiet)
    if full_output:
        tck, fp, ier, msg = res
        return (rp.array(tck[0]), rp.array(tck[1]), tck[2]), fp, ier, msg
    else:
        tck = res
        return (rp.array(tck[0]), rp.array(tck[1]), tck[2])
    
class _TorchRegularGridInterpolator:
    def __init__(self, points, values, method='linear', bounds_error=False, fill_value=np.nan):
        self.points = [tr.as_tensor(p) for p in points]
        self.values = tr.as_tensor(values)
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        
        if self.method != 'linear':
            raise NotImplementedError("Only 'linear' method is supported for PyTorch RegularGridInterpolator")
            
    def __call__(self, xi):
        xi = tr.as_tensor(xi)
        ndim = len(self.points)
        
        if xi.shape[-1] != ndim:
            raise ValueError(f"The requested sample points xi have dimension {xi.shape[-1]}, but this RegularGridInterpolator has dimension {ndim}")

        original_shape = xi.shape[:-1]
        xi_flat = xi.reshape(-1, ndim)
        n_points = xi_flat.shape[0]
        
        out_of_bounds = tr.zeros(n_points, dtype=tr.bool, device=xi.device)
        indices = []
        norm_weights = []
        
        for i, p in enumerate(self.points):
            out_of_bounds |= (xi_flat[:, i] < p[0]) | (xi_flat[:, i] > p[-1])
            
            idx = tr.searchsorted(p, xi_flat[:, i], right=True) - 1
            idx = tr.clamp(idx, 0, len(p) - 2)
            indices.append(idx)
            
            p_lo = p[idx]
            p_hi = p[idx + 1]
            
            dp = p_hi - p_lo
            w = tr.where(dp > 0, (xi_flat[:, i] - p_lo) / dp, tr.zeros_like(dp))
            if self.fill_value is not None:
                w = tr.clamp(w, 0.0, 1.0)
            norm_weights.append(w)
            
        if self.bounds_error and out_of_bounds.any():
            raise ValueError("One of the requested xi is out of bounds")
            
        import itertools
        corners = list(itertools.product([0, 1], repeat=ndim))
        
        res_shape = (n_points,) + self.values.shape[ndim:]
        res = tr.zeros(res_shape, dtype=self.values.dtype, device=self.values.device)
        
        for corner in corners:
            corner_indices = [indices[i] + corner[i] for i in range(ndim)]
            corner_vals = self.values[tuple(corner_indices)]
            
            corner_weight = tr.ones(n_points, dtype=xi_flat.dtype, device=xi.device)
            for i in range(ndim):
                corner_weight *= norm_weights[i] if corner[i] == 1 else (1.0 - norm_weights[i])
                
            while corner_weight.ndim < corner_vals.ndim:
                corner_weight = corner_weight.unsqueeze(-1)
                
            corner_weight = corner_weight.to(corner_vals.dtype)
            res += corner_vals * corner_weight
            
        if not self.bounds_error and self.fill_value is not None:
            oob_mask = out_of_bounds
            while oob_mask.ndim < res.ndim:
                oob_mask = oob_mask.unsqueeze(-1)
            fill_val_tensor = tr.tensor(self.fill_value, dtype=res.dtype, device=res.device)
            res = tr.where(oob_mask, fill_val_tensor, res)
            
        return TorchArray(res.reshape(*original_shape, *res.shape[1:]))
