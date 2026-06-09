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

jo = None
if j is not None:
    try:
        import jax.scipy.optimize as jso
        jo = jso
    except ImportError:
        pass

so  = sp.optimize if sp is not None else None

# ----------------------------------------------------------------------------------------------------------------------
#  Functions
# ----------------------------------------------------------------------------------------------------------------------  

def minimize(fun, x0, args=(), *, method='BFGS', bounds=None, constraints=(), tol=None, options=None): 
    if not isinstance(args, tuple):
        args = (args,)
    if rp.use_jax: 
        if bounds is not None:
             raise NotImplementedError('bounds are not supported for minimize in JAX')
        if constraints and len(constraints) > 0:
             raise NotImplementedError('constraints are not supported for minimize in JAX')
        return jo.minimize(fun, x0, args=args, method=method, tol=tol, options=options)
    elif rp.use_torch: 
        import torch as tr
        
        # Normalize constraints
        cons_list = []
        if isinstance(constraints, dict):
            cons_list = [constraints]
        elif isinstance(constraints, (list, tuple)):
            cons_list = list(constraints)
        
        for c in cons_list:
            if 'args' in c and not isinstance(c['args'], tuple):
                c['args'] = (c['args'],)

        # Collect all tensors from args, bounds, and constraints
        all_inputs = [args, bounds, cons_list]
        params = _find_tensors(all_inputs)
        dt = x0.dtype if hasattr(x0, 'dtype') else (params[0].dtype if params else tr.get_default_dtype())

        class Minimize(tr.autograd.Function):
            @staticmethod
            def forward(ctx, x0_in, *params_in):
                def get_current_inputs(p_in):
                    return _replace_tensors(all_inputs, p_in, {'idx': 0})
                
                def fun_np(x_val):
                    curr_args, curr_bounds, curr_cons = get_current_inputs(params_in)
                    res_val = fun(rp.array(x_val, dtype=dt), *curr_args)
                    return tr.as_tensor(res_val).detach().cpu().numpy()

                so_cons = []
                curr_all = get_current_inputs(params_in)
                curr_cons_list = curr_all[2]
                
                for c_idx, c in enumerate(curr_cons_list):
                    def c_np(x_v, idx=c_idx):
                        c_curr_all = get_current_inputs(params_in)
                        c_curr = c_curr_all[2][idx]
                        c_args = c_curr.get('args', ())
                        val = c_curr['fun'](rp.array(x_v, dtype=dt), *c_args)
                        return tr.as_tensor(val).detach().cpu().numpy()
                    so_cons.append({'type': c['type'], 'fun': c_np})

                curr_bounds = curr_all[1]
                res = so.minimize(fun_np, tr.as_tensor(x0_in).detach().cpu().numpy(), method=method, bounds=curr_bounds, constraints=so_cons, tol=tol, options=options)
                x_sol = tr.as_tensor(res.x, dtype=dt)
                
                # Identify Active Constraints and Multipliers
                multipliers = getattr(res, 'multipliers', None)
                active_cons_indices = []
                active_multipliers = []
                
                if multipliers is not None:
                    meq = sum(1 for c in cons_list if c['type'] == 'eq')
                    eq_mults = multipliers[:meq]
                    ineq_mults = multipliers[meq:]
                    ei, ii = 0, 0
                    for c_idx, c in enumerate(cons_list):
                        if c['type'] == 'eq':
                            active_cons_indices.append(c_idx)
                            active_multipliers.append(float(eq_mults[ei]))
                            ei += 1
                        else:
                            if ineq_mults[ii] > 1e-8:
                                active_cons_indices.append(c_idx)
                                active_multipliers.append(float(ineq_mults[ii]))
                            ii += 1
                else:
                    # Fallback identification by value
                    curr_all_fb = get_current_inputs(params_in)
                    curr_cons_fb = curr_all_fb[2]
                    for c_idx, c in enumerate(curr_cons_fb):
                        c_args = c.get('args', ())
                        val = c['fun'](rp.array(res.x, dtype=dt), *c_args)
                        if c['type'] == 'eq' or abs(val) < 1e-6:
                            active_cons_indices.append(c_idx)
                            active_multipliers.append(1.0) # Dummy

                active_bounds = []
                if bounds is not None:
                    res_x_np = res.x
                    for i, (l, u) in enumerate(bounds):
                        if l is not None and abs(res_x_np[i] - l) < 1e-7:
                            active_bounds.append((i, float(l), -1.0))
                        elif u is not None and abs(res_x_np[i] - u) < 1e-7:
                            active_bounds.append((i, float(u), 1.0))

                ctx.save_for_backward(x_sol, *params_in)
                ctx.active_info = (active_cons_indices, tr.tensor(active_multipliers, dtype=dt), active_bounds)
                return x_sol

            @staticmethod
            def backward(ctx, grad_x):
                x_sol = ctx.saved_tensors[0]
                params_in = ctx.saved_tensors[1:]
                active_cons_indices, active_multipliers, active_bounds = ctx.active_info
                
                def get_inputs(p_in):
                    return _replace_tensors(all_inputs, p_in, {'idx': 0})

                with tr.enable_grad():
                    # Re-enable gradients for params_in to allow VJP computation
                    diff_params = [p.detach().requires_grad_(True) for p in params_in]
                    x = x_sol.detach().requires_grad_(True)
                    
                    # Compute Lagrangian and its gradient w.r.t x
                    def eval_lagrangian(x_v, p_v):
                         curr_args, curr_bounds, curr_cons_list = get_inputs(p_v)
                         f_v = fun(rp.array(x_v, dtype=dt), *curr_args)
                         L_val = tr.as_tensor(f_v).sum()
                         
                         for m_idx, c_idx in enumerate(active_cons_indices):
                              c_curr = curr_cons_list[c_idx]
                              c_a = c_curr.get('args', ())
                              cv = c_curr['fun'](rp.array(x_v, dtype=dt), *c_a)
                              L_val = L_val - active_multipliers[m_idx] * tr.as_tensor(cv).sum()
                         
                         for b_i, b_v, b_d in active_bounds:
                              # Multiplier for bounds (estimated as 1.0 if not available)
                              m_b = active_multipliers[len(active_cons_indices) + active_bounds.index((b_i, b_v, b_d))] if len(active_multipliers) > len(active_cons_indices) else 1.0
                              if b_d == -1.0: L_val = L_val - m_b * (x_v[b_i] - b_v)
                              else: L_val = L_val - m_b * (b_v - x_v[b_i])
                         return L_val

                    lagrangian = eval_lagrangian(x, diff_params)
                    tr.autograd.grad(lagrangian, x, create_graph=True)[0]
                    
                    # c_active(x, p) for the KKT system
                    c_active_vals = []
                    curr_args, curr_bounds, curr_cons_list = get_inputs(diff_params)
                    for c_idx in active_cons_indices:
                        c_curr = curr_cons_list[c_idx]
                        c_a = c_curr.get('args', ())
                        cv = c_curr['fun'](rp.array(x, dtype=dt), *c_a)
                        c_active_vals.append(tr.as_tensor(cv).sum())
                    for b_i, b_v, b_d in active_bounds:
                        if b_d == -1.0: c_active_vals.append(x[b_i] - b_v)
                        else: c_active_vals.append(b_v - x[b_i])

                    # 2. Form KKT Matrix components
                    # H = \nabla_xx L
                    def grad_L_x_pure_x(x_v):
                         return tr.autograd.grad(eval_lagrangian(x_v, diff_params), x_v, create_graph=True)[0]
                    H = tr.autograd.functional.jacobian(grad_L_x_pure_x, x)
                    
                    # A = \nabla_x c_active
                    if len(c_active_vals) > 0:
                        def c_stack_pure_x(x_v):
                             ga, gb, gc_list = get_inputs(diff_params)
                             cvs = []
                             for ci in active_cons_indices:
                                  cc = gc_list[ci]
                                  ca = cc.get('args', ())
                                  cvs.append(tr.as_tensor(cc['fun'](rp.array(x_v, dtype=dt), *ca)).sum())
                             for bi, bv, bd in active_bounds:
                                  if bd == -1.0: cvs.append(x_v[bi] - bv)
                                  else: cvs.append(bv - x_v[bi])
                             return tr.stack(cvs)
                        A = tr.autograd.functional.jacobian(c_stack_pure_x, x)
                    else:
                        A = tr.zeros((0, x.shape[0]), dtype=dt)

                # Solve KKT adjoint: [H A^T; A 0] [v_x; v_lam] = [grad_x; 0]
                n_x, n_c = x.shape[0], A.shape[0]
                KKT = tr.zeros((n_x + n_c, n_x + n_c), dtype=dt)
                KKT[:n_x, :n_x] = H.detach()
                KKT[:n_x, n_x:] = A.detach().T
                KKT[n_x:, :n_x] = A.detach()
                rhs = tr.zeros(n_x + n_c, dtype=dt)
                rhs[:n_x] = grad_x.detach()
                try:    sol = tr.linalg.solve(KKT, rhs.unsqueeze(-1)).squeeze(-1)
                except: sol = tr.linalg.lstsq(KKT, rhs.unsqueeze(-1)).solution.squeeze(-1)
                v_x  = sol[:n_x].detach()
                v_lam = sol[n_x:].detach()

                # IFT parameter gradient:
                #   dL_loss/dp = -v_x . d(grad_L_x)/dp  -  v_lam . d(c_active)/dp
                # where x is fixed at x* and we differentiate w.r.t. parameters p only.
                # We build a scalar sensitivity = v_x . grad_L_x(x*, p) + v_lam . c(x*, p)
                # and take -d(sensitivity)/dp.
                grad_params = [None] * len(params_in)
                params_req_grad = [p for p in params_in if p.requires_grad]
                params_indices  = [i for i, p in enumerate(params_in) if p.requires_grad]

                if params_req_grad:
                    with tr.enable_grad():
                        # Fresh diff_params that require grad
                        dp = [p.detach().requires_grad_(p.requires_grad) for p in params_in]
                        x_star = x_sol.detach().requires_grad_(True)  # need grad for d L/dx

                        # Lagrangian(x*, dp) -- both x and dp participate in graph
                        L_xp = eval_lagrangian(x_star, dp)

                        # grad_L_x evaluated at (x*, dp) -- function of dp through L_xp
                        gL_x = tr.autograd.grad(L_xp, x_star, create_graph=True)[0]
                        # gL_x depends on dp through eval_lagrangian → create_graph=True keeps the graph

                        # Sensitivity scalar:  v_x . gL_x(x*, p)
                        sensitivity = (v_x * gL_x).sum()

                        # Add: v_lam . c_active(x*, p)  -- constraints depend on dp
                        if v_lam.numel() > 0:
                            _, _, c_list_dp = get_inputs(dp)
                            for m_i, c_idx in enumerate(active_cons_indices):
                                cc = c_list_dp[c_idx]
                                ca = cc.get('args', ())
                                cv = tr.as_tensor(cc['fun'](rp.array(x_sol.detach(), dtype=dt), *ca), dtype=dt).reshape([])
                                sensitivity = sensitivity + v_lam[m_i] * cv

                        dp_req = [p for p in dp if p.requires_grad]
                        if dp_req and sensitivity.requires_grad:
                            grads = tr.autograd.grad(sensitivity, dp_req, allow_unused=True)
                            j = 0
                            for i in params_indices:
                                g = grads[j] if j < len(grads) else None
                                grad_params[i] = -g if g is not None else tr.zeros_like(params_in[i])
                                j += 1
                        elif dp_req:
                            # sensitivity has no graph -- function truly independent of params
                            for i in params_indices:
                                grad_params[i] = tr.zeros_like(params_in[i])

                return (None, *grad_params)

        res_x = Minimize.apply(x0, *params)
        
        def fun_np_meta(x_v):
             res_val = fun(rp.array(x_v, dtype=dt), *_replace_tensors(args, params, {'idx': 0}))
             return tr.as_tensor(res_val).detach().cpu().numpy()
        
        # Metadata retrieval
        meta_cons = []
        curr_all_meta = _replace_tensors(all_inputs, params, {'idx': 0})
        curr_cons_meta = curr_all_meta[2]
        for c_idx, c in enumerate(curr_cons_meta):
             def c_meta_np(x_v, idx=c_idx):
                  m_curr_all = _replace_tensors(all_inputs, params, {'idx': 0})
                  m_c = m_curr_all[2][idx]
                  m_a = m_c.get('args', ())
                  val = m_c['fun'](rp.array(x_v, dtype=dt), *m_a)
                  return tr.as_tensor(val).detach().cpu().numpy()
             meta_cons.append({'type': c['type'], 'fun': c_meta_np})
             
        res = so.minimize(fun_np_meta, tr.as_tensor(x0).detach().cpu().numpy(), method=method, bounds=curr_all_meta[1], constraints=meta_cons, tol=tol, options=options)
        res = _convert_optimize_result(res)
        res.x = rp.array(res_x, dtype=dt)
        return res

    else: return so.minimize(fun, x0, args=args, method=method, bounds=bounds, constraints=constraints, tol=tol, options=options)

def fmin_slsqp(func, x0, fprime=None, f_eqcons=None, fprime_eqcons=None, 
               f_ieqcons=None, fprime_ieqcons=None, bounds=(), iter=100, 
               acc=1e-06, iprint=1, disp=None, full_output=0, 
               epsilon=1.4901161193847656e-08, callback=None, args=()): 
    if not isinstance(args, tuple):
        args = (args,)
    
    

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

def _find_tensors(obj):
    import torch as tr
    tensors = []
    if isinstance(obj, tr.Tensor):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_find_tensors(item))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            tensors.extend(_find_tensors(v))
    elif hasattr(obj, 'items'): 
        for k, v in obj.items():
            tensors.extend(_find_tensors(v))
    return tensors

def _replace_tensors(obj, tensors, state):
    import torch as tr
    if isinstance(obj, tr.Tensor):
        val = tensors[state['idx']]
        state['idx'] += 1
        return val
    elif type(obj) is list:
        return [_replace_tensors(item, tensors, state) for item in obj]
    elif type(obj) is tuple:
        return tuple(_replace_tensors(item, tensors, state) for item in obj)
    elif type(obj) is dict:
        return {k: _replace_tensors(v, tensors, state) for k, v in obj.items()}
    elif hasattr(obj, 'items') and hasattr(obj, 'copy'):
        import copy
        new_obj = copy.copy(obj)
        for k, v in obj.items():
            new_obj[k] = _replace_tensors(v, tensors, state)
        return new_obj
    return obj

def _convert_optimize_result(res):
    """Recursively converts NumPy arrays in an OptimizeResult or dict to RNUMPY arrays."""
    if isinstance(res, dict) or hasattr(res, 'items'):
        for key, value in res.items():
             if isinstance(value, (np.ndarray, np.generic)):
                  res[key] = rp.array(value)
             elif isinstance(value, (dict)) or hasattr(value, 'items'):
                  _convert_optimize_result(value)
    return res

def minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None, tol=None, options=None): 
    if not isinstance(args, tuple):
        args = (args,)
    if rp.use_jax:
        import jax
        import jax.numpy as jnp
        
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        
        @jax.custom_jvp
        def _jax_solve(bracket_in, bounds_in, *a_flat):
            current_args = jax.tree_util.tree_unflatten(args_tree, a_flat)
            def fun_np(x_val):
                 res = fun(rp.array(x_val), *current_args)
                 return np.asarray(jax.device_get(res))
            
            res = so.minimize_scalar(fun_np, bracket=bracket_in, bounds=bounds_in, method=method, tol=tol, options=options)
            return jnp.array(res.x), jnp.array(res.fun)

        @_jax_solve.defjvp
        def _jax_solve_jvp(primals, tangents):
            bracket_p, bounds_p = primals[:2]
            params_p = primals[2:]
            params_t = tangents[2:]
            
            x_sol, f_sol = _jax_solve(*primals)
            
            # IFT: f'(x, p) = 0  => dx/dp = - (d^2 f / dx dp) / (d^2 f / dx^2)
            def objective(x_v, *p_v):
                current_args = jax.tree_util.tree_unflatten(args_tree, p_v)
                return fun(rp.array(x_v), *current_args).sum()

            f_prime_v_x = jax.grad(objective, argnums=0)
            f_pp_v_x = jax.grad(f_prime_v_x, argnums=0)(x_sol, *params_p)
            
            # JVP for f_prime at (x_sol, params)
            _, f_prime_tangent = jax.jvp(lambda *p: f_prime_v_x(x_sol, *p), params_p, params_t)
            dx = -f_prime_tangent / f_pp_v_x
            
            # Envelope theorem: df/dp = partial_f/partial_p at fixed x_sol
            _, df_dp_partial = jax.jvp(lambda *p: objective(x_sol, *p), params_p, params_t)
            return (x_sol, f_sol), (dx, df_dp_partial)

        x_val, f_val = _jax_solve(bracket, bounds, *args_flat)
        
        # To get the full OptimizeResult object (metadata like success, nit), 
        # we run it once with stop_gradient to avoid tracer errors.
        def get_metadata():
            def fun_np_meta(x_v):
                # Use stop_gradient to get concrete values for the metadata solver
                res_v = fun(rp.array(x_v), *jax.tree_util.tree_map(jax.lax.stop_gradient, args))
                return np.asarray(jax.device_get(res_v))
            return so.minimize_scalar(fun_np_meta, bracket=bracket, bounds=bounds, method=method, tol=tol, options=options)
            
        try:
            res = get_metadata()
            res = _convert_optimize_result(res)
        except Exception:
            # Fallback if metadata retrieval fails
            from scipy.optimize import OptimizeResult
            res = OptimizeResult(x=x_val, fun=f_val, success=True, status=0, message='Success', nit=0, nfev=0)

        res.x = x_val
        res.fun = f_val
        return res

    elif rp.use_torch:
        import torch as tr
        
        params = _find_tensors(args)
        dt = params[0].dtype if params else tr.get_default_dtype()
        
        class MinimizeScalar(tr.autograd.Function):
            @staticmethod
            def forward(ctx, *params_in):
                bracket, bounds = params_in[0], params_in[1]
                args_tensors = params_in[2:]
                
                def fun_np(x_val):
                    current_args = _replace_tensors(args, args_tensors, {'idx': 0})
                    res = fun(rp.array(x_val, dtype=dt), *current_args)
                    return tr.as_tensor(res).detach().cpu().numpy()

                res = so.minimize_scalar(fun_np, bracket=bracket, bounds=bounds, method=method, tol=tol, options=options)
                x_sol = tr.as_tensor(res.x, dtype=dt) 
                ctx.save_for_backward(x_sol, *args_tensors)
                return x_sol, tr.as_tensor(res.fun, dtype=dt)

            @staticmethod
            def backward(ctx, grad_x, grad_f):
                x_sol = ctx.saved_tensors[0]
                params_in = ctx.saved_tensors[1:]
                
                with tr.enable_grad():
                    x = x_sol.detach().requires_grad_(True)
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    f = fun(rp.array(x, dtype=dt), *current_args)
                    f_tensor = tr.as_tensor(f).sum()
                    
                    # f_prime = df/dx
                    f_prime = tr.autograd.grad(f_tensor, x, create_graph=True)[0]
                    # f_double_prime = d^2 f / dx^2
                    f_double_prime = tr.autograd.grad(f_prime, x, retain_graph=True)[0]
                
                # IFT: dx/dp = - (1/f'') * (df'/dp)
                multiplier = - grad_x / f_double_prime
                
                grad_params = [None] * len(params_in)
                params_to_diff = []
                params_indices = []
                for i, p in enumerate(params_in):
                    if p.requires_grad:
                        params_to_diff.append(p)
                        params_indices.append(i)
                
                if params_to_diff:
                    # G_x = grad_x * dx/dp
                    vjp_x = tr.autograd.grad(f_prime, params_to_diff, grad_outputs=multiplier, retain_graph=True, allow_unused=True)
                    # G_f = grad_f * df/dp (direct)
                    vjp_f = tr.autograd.grad(f_tensor, params_to_diff, grad_outputs=grad_f.expand_as(f_tensor), allow_unused=True)
                    
                    for i, (gx, gf) in zip(params_indices, zip(vjp_x, vjp_f)):
                        g = 0
                        if gx is not None: g = g + gx
                        if gf is not None: g = g + gf
                        grad_params[i] = g
                
                return (None, None, *grad_params)

        params = _find_tensors(args)
        x_val, f_val = MinimizeScalar.apply(bracket, bounds, *params)
        
        # OptimizeResult
        def fun_np_final(x_val): 
             res = fun(rp.array(x_val, dtype=dt), *_replace_tensors(args, params, {'idx': 0}))
             return tr.as_tensor(res).detach().cpu().numpy()
        res = so.minimize_scalar(fun_np_final, bracket=bracket, bounds=bounds, method=method, tol=tol, options=options)
        res = _convert_optimize_result(res)
        res.x = rp.array(x_val)
        res.fun = rp.array(f_val)
        return res
    
    else:
        return so.minimize_scalar(fun, bracket=bracket, bounds=bounds, args=args, method=method, tol=tol, options=options)

def brentq(f, a, b, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, full_output=False, disp=True):
    if not isinstance(args, tuple):
        args = (args,)
    if rp.use_jax:
        import jax
        import jax.numpy as jnp
        
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        
        @jax.custom_jvp
        def _jax_brentq(a_in, b_in, *a_flat):
            current_args = jax.tree_util.tree_unflatten(args_tree, a_flat)
            def fun_np(x_v): 
                res = f(rp.array(x_v), *current_args)
                return np.asarray(jax.device_get(res))
            
            x_sol = so.brentq(fun_np, a_in, b_in, args=(), xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp)
            return jnp.array(x_sol)

        @_jax_brentq.defjvp
        def _jax_brentq_jvp(primals, tangents):
            params_p = primals[2:]
            params_t = tangents[2:]
            x_sol = _jax_brentq(*primals)
            
            def objective(x_v, *p_v):
                current_args = jax.tree_util.tree_unflatten(args_tree, p_v)
                return f(rp.array(x_v), *current_args).sum()

            df_dx = jax.grad(objective, argnums=0)(x_sol, *params_p)
            _, df_dp_tangent = jax.jvp(lambda *p: objective(x_sol, *p), params_p, params_t)
            
            dx = -df_dp_tangent / df_dx
            return x_sol, dx

        x_sol = _jax_brentq(a, b, *args_flat)
        
        if full_output:
            def get_metadata():
                def fun_np_meta(x_v):
                    res_v = f(rp.array(x_v), *jax.tree_util.tree_map(jax.lax.stop_gradient, args))
                    return np.asarray(jax.device_get(res_v))
                return so.brentq(fun_np_meta, a, b, args=(), xtol=xtol, rtol=rtol, maxiter=maxiter, full_output=True, disp=disp)
            xr, r = get_metadata()
            return x_sol, r
        else:
            return x_sol

    elif rp.use_torch:
        import torch as tr
        
        params = _find_tensors(args)
        dt = params[0].dtype if params else tr.get_default_dtype()

        class BrentQ(tr.autograd.Function):
            @staticmethod
            def forward(ctx, a_in, b_in, *params_in):
                def fun_np(x_val):
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    res = f(rp.array(x_val, dtype=dt), *current_args)
                    return tr.as_tensor(res).detach().cpu().numpy()

                x_sol = so.brentq(fun_np, a_in, b_in, args=(), xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp)
                x_sol_t = tr.as_tensor(x_sol, dtype=dt)
                ctx.save_for_backward(x_sol_t, *params_in)
                return x_sol_t

            @staticmethod
            def backward(ctx, grad_x):
                x_sol = ctx.saved_tensors[0]
                params_in = ctx.saved_tensors[1:]
                
                with tr.enable_grad():
                    x = x_sol.detach().requires_grad_(True)
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    val = f(rp.array(x, dtype=dt), *current_args)
                    f_tensor = tr.as_tensor(val).sum()
                    df_dx = tr.autograd.grad(f_tensor, x, retain_graph=True)[0]
                
                multiplier = - grad_x / df_dx
                grad_params = [None] * len(params_in)
                params_to_diff = []
                params_indices = []
                for i, p in enumerate(params_in):
                    if p.requires_grad:
                        params_to_diff.append(p)
                        params_indices.append(i)
                
                if params_to_diff:
                    grads_p = tr.autograd.grad(f_tensor, params_to_diff, grad_outputs=multiplier.expand_as(f_tensor), allow_unused=True)
                    for i, g in zip(params_indices, grads_p):
                        grad_params[i] = g
                
                return (None, None, *grad_params)

        params = _find_tensors(args)
        x_sol = BrentQ.apply(a, b, *params)
        
        if full_output:
            def fun_np_meta(x_val):
                 res = f(rp.array(x_val, dtype=dt), *_replace_tensors(args, params, {'idx': 0}))
                 return tr.as_tensor(res).detach().cpu().numpy()
            xr, r = so.brentq(fun_np_meta, a, b, args=(), xtol=xtol, rtol=rtol, maxiter=maxiter, full_output=True, disp=disp)
            return rp.array(x_sol, dtype=dt), r
        else:
            return rp.array(x_sol, dtype=dt)
    
    else:
        return so.brentq(f, a, b, args=args, xtol=xtol, rtol=rtol, maxiter=maxiter, full_output=full_output, disp=disp)

def fminbound(func, x1, x2, args=(), xtol=1e-05, maxfun=500, full_output=0, disp=1):
    if not isinstance(args, tuple):
        args = (args,)
    if rp.use_jax:
        import jax
        import jax.numpy as jnp
        
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        
        @jax.custom_jvp
        def _jax_fminbound(x1_in, x2_in, *a_flat):
            current_args = jax.tree_util.tree_unflatten(args_tree, a_flat)
            def fun_np(x_v): 
                res = func(rp.array(x_v), *current_args)
                return np.asarray(jax.device_get(res))
            
            x_sol = so.fminbound(fun_np, x1_in, x2_in, args=(), xtol=xtol, maxfun=maxfun, disp=disp)
            return jnp.array(x_sol)

        @_jax_fminbound.defjvp
        def _jax_fminbound_jvp(primals, tangents):
            params_p = primals[2:]
            params_t = tangents[2:]
            x_sol = _jax_fminbound(*primals)
            
            def objective(x_v, *p_v):
                current_args = jax.tree_util.tree_unflatten(args_tree, p_v)
                return func(rp.array(x_v), *current_args).sum()

            f_prime = jax.grad(objective, argnums=0)
            f_pp = jax.grad(f_prime, argnums=0)(x_sol, *params_p)
            _, f_prime_tangent = jax.jvp(lambda *p: f_prime(x_sol, *p), params_p, params_t)
            
            dx = -f_prime_tangent / f_pp
            return x_sol, dx

        x_sol = _jax_fminbound(x1, x2, *args_flat)
        
        if full_output:
            def get_metadata():
                def fun_np_meta(x_v):
                    res_v = func(rp.array(x_v), *jax.tree_util.tree_map(jax.lax.stop_gradient, args))
                    return np.asarray(jax.device_get(res_v))
                return so.fminbound(fun_np_meta, x1, x2, args=(), xtol=xtol, maxfun=maxfun, full_output=True, disp=disp)
            xr, fval, ierr, numfunc = get_metadata()
            return x_sol, fval, ierr, numfunc
        else:
            return x_sol

    elif rp.use_torch:
        import torch as tr
        
        params = _find_tensors(args)
        dt = params[0].dtype if params else tr.get_default_dtype()

        class FMinBound(tr.autograd.Function):
            @staticmethod
            def forward(ctx, x1_in, x2_in, *params_in):
                def fun_np(x_val):
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    res = func(rp.array(x_val, dtype=dt), *current_args)
                    return tr.as_tensor(res).detach().cpu().numpy()

                x_sol = so.fminbound(fun_np, x1_in, x2_in, args=(), xtol=xtol, maxfun=maxfun, disp=disp)
                x_sol_t = tr.as_tensor(x_sol, dtype=dt)
                ctx.save_for_backward(x_sol_t, *params_in)
                return x_sol_t

            @staticmethod
            def backward(ctx, grad_x):
                x_sol = ctx.saved_tensors[0]
                params_in = ctx.saved_tensors[1:]
                
                with tr.enable_grad():
                    x = x_sol.detach().requires_grad_(True)
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    val = func(rp.array(x, dtype=dt), *current_args)
                    f_tensor = tr.as_tensor(val).sum()
                    f_prime = tr.autograd.grad(f_tensor, x, create_graph=True)[0]
                    f_double_prime = tr.autograd.grad(f_prime, x, retain_graph=True)[0]
                
                multiplier = - grad_x / f_double_prime
                grad_params = [None] * len(params_in)
                params_to_diff = []
                params_indices = []
                for i, p in enumerate(params_in):
                    if p.requires_grad:
                        params_to_diff.append(p)
                        params_indices.append(i)
                
                if params_to_diff:
                    grads_p = tr.autograd.grad(f_prime, params_to_diff, grad_outputs=multiplier.expand_as(f_prime), allow_unused=True)
                    for i, g in zip(params_indices, grads_p):
                        grad_params[i] = g
                
                return (None, None, *grad_params)

        params = _find_tensors(args)
        x_sol = FMinBound.apply(x1, x2, *params)
        
        if full_output:
            def fun_np_meta(x_val):
                 res = func(rp.array(x_val, dtype=dt), *_replace_tensors(args, params, {'idx': 0}))
                 return tr.as_tensor(res).detach().cpu().numpy()
            xr, fval, ierr, numfunc = so.fminbound(fun_np_meta, x1, x2, args=(), xtol=xtol, maxfun=maxfun, full_output=True, disp=disp)
            return rp.array(x_sol, dtype=dt), fval, ierr, numfunc
        else:
            return rp.array(x_sol, dtype=dt)
    
    else:
        return so.fminbound(func, x1, x2, args=args, xtol=xtol, maxfun=maxfun, full_output=full_output, disp=disp)

def fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None):
    if not isinstance(args, tuple):
        args = (args,)
    if rp.use_jax:
        import jax
        import jax.numpy as jnp
        
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        
        @jax.custom_jvp
        def _jax_fsolve(x0_in, *a_flat):
            current_args = jax.tree_util.tree_unflatten(args_tree, a_flat)
            def fun_np(x_v): 
                 res = func(rp.array(x_v), *current_args)
                 return np.asarray(jax.device_get(res))
            
            fprime_to_use = None
            if fprime is not None:
                def fprime_np(x_v):
                    res = fprime(rp.array(x_v), *current_args)
                    return np.asarray(jax.device_get(res))
                fprime_to_use = fprime_np
            else:
                def fprime_autograd_np(x_v):
                    jac_fn = jax.jacobian(lambda x: func(rp.array(x), *current_args).ravel())
                    res = jac_fn(jnp.array(x_v))
                    return np.asarray(jax.device_get(res))
                fprime_to_use = fprime_autograd_np
            
            res = so.fsolve(fun_np, np.asarray(x0_in), args=(), fprime=fprime_to_use, full_output=True, 
                            col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, band=band, 
                            epsfcn=epsfcn, factor=factor, diag=diag)
            x_sol, infodict, ier, mesg = res
            return jnp.array(x_sol)

        @_jax_fsolve.defjvp
        def _jax_fsolve_jvp(primals, tangents):
            params_p = primals[1:]
            params_t = tangents[1:]
            x_sol = _jax_fsolve(*primals)
            
            def objective(x_v, *p_v):
                current_args = jax.tree_util.tree_unflatten(args_tree, p_v)
                return func(rp.array(x_v), *current_args).ravel()

            Jx = jax.jacobian(objective, argnums=0)(x_sol, *params_p)
            _ , f_p_tangent = jax.jvp(lambda *p: objective(x_sol, *p), params_p, params_t)
            
            dx = -jnp.linalg.solve(Jx, f_p_tangent)
            return x_sol, dx

        x_sol = _jax_fsolve(x0, *args_flat)
        
        if full_output:
            def get_metadata():
                def fun_np_meta(x_v):
                    res_v = func(rp.array(x_v), *jax.tree_util.tree_map(jax.lax.stop_gradient, args))
                    return np.asarray(jax.device_get(res_v))
                
                fprime_to_use_meta = None
                if fprime is not None:
                    def fprime_np_meta(x_v):
                        res_v = fprime(rp.array(x_v), *jax.tree_util.tree_map(jax.lax.stop_gradient, args))
                        return np.asarray(jax.device_get(res_v))
                    fprime_to_use_meta = fprime_np_meta
                else:
                    def fprime_autograd_np_meta(x_v):
                        jac_fn = jax.jacobian(lambda x: func(rp.array(x), *jax.tree_util.tree_map(jax.lax.stop_gradient, args)).ravel())
                        res = jac_fn(jnp.array(x_v))
                        return np.asarray(jax.device_get(res))
                    fprime_to_use_meta = fprime_autograd_np_meta

                return so.fsolve(fun_np_meta, np.asarray(x0), args=(), fprime=fprime_to_use_meta, full_output=True, 
                                 col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, band=band, 
                                 epsfcn=epsfcn, factor=factor, diag=diag)
            
            try:
                xs, infodict, ier, mesg = get_metadata()
                return rp.array(x_sol), _convert_optimize_result(infodict), ier, mesg
            except Exception:
                return rp.array(x_sol), {}, 1, "Success"
        else:
            return rp.array(x_sol)

    elif rp.use_torch:
        import torch as tr
        
        params = _find_tensors(args)
        dt = x0.dtype if hasattr(x0, 'dtype') else (params[0].dtype if params else tr.get_default_dtype())

        class FSolve(tr.autograd.Function):
            @staticmethod
            def forward(ctx, x0_in, *params_in):
                def fun_np(x_val):
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    res = func(rp.array(x_val, dtype=dt), *current_args)
                    return tr.as_tensor(res).detach().cpu().numpy()

                fprime_to_use = None
                if fprime is not None:
                    def fprime_np(x_val):
                        current_args = _replace_tensors(args, params_in, {'idx': 0})
                        res = fprime(rp.array(x_val, dtype=dt), *current_args)
                        return tr.as_tensor(res).detach().cpu().numpy()
                    fprime_to_use = fprime_np
                else:
                    def fprime_autograd_np(x_val):
                        current_args = _replace_tensors(args, params_in, {'idx': 0})
                        x_tr = tr.as_tensor(x_val, dtype=dt)
                        def f_tr(x_v):
                            return tr.as_tensor(func(rp.array(x_v, dtype=dt), *current_args)).ravel()
                        jac = tr.autograd.functional.jacobian(f_tr, x_tr)
                        return jac.detach().cpu().numpy()
                    fprime_to_use = fprime_autograd_np

                res = so.fsolve(fun_np, tr.as_tensor(x0_in).detach().cpu().numpy(), args=(), fprime=fprime_to_use, 
                                 full_output=True, col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, 
                                 band=band, epsfcn=epsfcn, factor=factor, diag=diag)
                x_sol_np, infodict, ier, mesg = res
                x_sol = tr.as_tensor(x_sol_np, dtype=dt)
                ctx.save_for_backward(x_sol, *params_in)
                return x_sol

            @staticmethod
            def backward(ctx, grad_x):
                x_sol = ctx.saved_tensors[0]
                params_in = ctx.saved_tensors[1:]
                
                with tr.enable_grad():
                    x = x_sol.detach().requires_grad_(True)
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    f_val = func(rp.array(x, dtype=dt), *current_args)
                    f_tensor = tr.as_tensor(f_val).ravel()
                
                def f_wrapper(x_v):
                    return tr.as_tensor(func(rp.array(x_v, dtype=dt), *current_args)).ravel()
                
                Jx = tr.autograd.functional.jacobian(f_wrapper, x)
                
                try:
                    lambd = tr.linalg.solve(Jx.T, grad_x.reshape(-1, 1)).reshape(-1)
                except:
                    lambd = tr.linalg.lstsq(Jx.T, grad_x.reshape(-1, 1)).solution.reshape(-1)
                
                grad_params = [None] * len(params_in)
                params_to_diff = []
                params_indices = []
                for i, p in enumerate(params_in):
                    if p.requires_grad:
                        params_to_diff.append(p)
                        params_indices.append(i)
                
                if params_to_diff:
                    vjp_params = tr.autograd.grad(f_tensor, params_to_diff, grad_outputs=-lambd, allow_unused=True)
                    for i, g in zip(params_indices, vjp_params):
                        grad_params[i] = g
                
                return (None, *grad_params)

        params = _find_tensors(args)
        x_sol = FSolve.apply(x0, *params)
        
        if full_output:
            def get_metadata():
                def fun_np_meta(x_val):
                    res = func(rp.array(x_val, dtype=dt), *_replace_tensors(args, params, {'idx': 0}))
                    return tr.as_tensor(res).detach().cpu().numpy()
                
                fprime_to_use_meta = None
                if fprime is not None:
                    def fprime_np_meta(x_val):
                        res = fprime(rp.array(x_val, dtype=dt), *_replace_tensors(args, params, {'idx': 0}))
                        return tr.as_tensor(res).detach().cpu().numpy()
                    fprime_to_use_meta = fprime_np_meta
                else:
                    def fprime_autograd_np_meta(x_val):
                        m_args = _replace_tensors(args, params, {'idx': 0})
                        x_tr = tr.as_tensor(x_val, dtype=dt)
                        def f_tr(x_v):
                            return tr.as_tensor(func(rp.array(x_v, dtype=dt), *m_args)).ravel()
                        jac = tr.autograd.functional.jacobian(f_tr, x_tr)
                        return jac.detach().cpu().numpy()
                    fprime_to_use_meta = fprime_autograd_np_meta

                return so.fsolve(fun_np_meta, tr.as_tensor(x0).detach().cpu().numpy(), args=(), fprime=fprime_to_use_meta, 
                                 full_output=True, col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, 
                                 band=band, epsfcn=epsfcn, factor=factor, diag=diag)
            
            xs, infodict, ier, mesg = get_metadata()
            return rp.array(x_sol, dtype=dt), _convert_optimize_result(infodict), ier, mesg
        else:
            return rp.array(x_sol, dtype=dt)
    
    else:
        res = so.fsolve(func, x0, args=args, fprime=fprime, full_output=full_output, 
                         col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, band=band, 
                         epsfcn=epsfcn, factor=factor, diag=diag)
        if full_output:
            x, infodict, ier, mesg = res
            return rp.array(x), _convert_optimize_result(infodict), ier, mesg
        else:
            return rp.array(res)

def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None):
    if not isinstance(args, tuple):
        args = (args,)
    if rp.use_jax:
        import jax
        import jax.numpy as jnp
        
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        
        @jax.custom_jvp
        def _jax_root(x0_in, *a_flat):
            current_args = jax.tree_util.tree_unflatten(args_tree, a_flat)
            def fun_np(x_v): 
                 res = fun(rp.array(x_v), *current_args)
                 return np.asarray(jax.device_get(res))
            
            res = so.root(fun_np, np.asarray(x0_in), method=method, tol=tol, callback=callback, options=options)
            return jnp.array(res.x)

        @_jax_root.defjvp
        def _jax_root_jvp(primals, tangents):
            params_p = primals[1:]
            params_t = tangents[1:]
            x_sol = _jax_root(*primals)
            
            def objective(x_v, *p_v):
                current_args = jax.tree_util.tree_unflatten(args_tree, p_v)
                return fun(rp.array(x_v), *current_args).ravel()

            Jx = jax.jacobian(objective, argnums=0)(x_sol, *params_p)
            _ , f_p_tangent = jax.jvp(lambda *p: objective(x_sol, *p), params_p, params_t)
            
            # Solve Jx * dx = -f_p_tangent
            dx = -jnp.linalg.solve(Jx, f_p_tangent)
            return x_sol, dx

        x_sol = _jax_root(x0, *args_flat)
        
        # Metadata retrieval
        def get_metadata():
            def fun_np_meta(x_v):
                res_v = fun(rp.array(x_v), *jax.tree_util.tree_map(jax.lax.stop_gradient, args))
                return np.asarray(jax.device_get(res_v))
            return so.root(fun_np_meta, np.asarray(x0), method=method, tol=tol, callback=callback, options=options)
            
        try:
            res = get_metadata()
            res = _convert_optimize_result(res)
        except Exception:
            from scipy.optimize import OptimizeResult
            res = OptimizeResult(x=x_sol, success=True, status=0, fun=x_sol*0, message='Success')

        res.x = x_sol
        return res

    elif rp.use_torch:
        import torch as tr
        
        params = _find_tensors(args)
        dt = x0.dtype if hasattr(x0, 'dtype') else (params[0].dtype if params else tr.get_default_dtype())

        class Root(tr.autograd.Function):
            @staticmethod
            def forward(ctx, x0_in, *params_in):
                def fun_np(x_val):
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    res = fun(rp.array(x_val, dtype=dt), *current_args)
                    return tr.as_tensor(res).detach().cpu().numpy()

                res = so.root(fun_np, tr.as_tensor(x0_in).detach().cpu().numpy(), method=method, tol=tol, callback=callback, options=options)
                x_sol = tr.as_tensor(res.x, dtype=dt)
                ctx.save_for_backward(x_sol, *params_in)
                return x_sol

            @staticmethod
            def backward(ctx, grad_x):
                x_sol = ctx.saved_tensors[0]
                params_in = ctx.saved_tensors[1:]
                
                with tr.enable_grad():
                    x = x_sol.detach().requires_grad_(True)
                    current_args = _replace_tensors(args, params_in, {'idx': 0})
                    f_val = fun(rp.array(x, dtype=dt), *current_args)
                    f_tensor = tr.as_tensor(f_val).ravel()
                
                def f_wrapper(x_v):
                    return tr.as_tensor(fun(rp.array(x_v, dtype=dt), *current_args)).ravel()
                
                Jx = tr.autograd.functional.jacobian(f_wrapper, x)
                
                # Solve Jx.T * lambda = grad_x
                try:
                    lambd = tr.linalg.solve(Jx.T, grad_x.reshape(-1, 1)).reshape(-1)
                except:
                    lambd = tr.linalg.lstsq(Jx.T, grad_x.reshape(-1, 1)).solution.reshape(-1)
                
                grad_params = [None] * len(params_in)
                params_to_diff = []
                params_indices = []
                for i, p in enumerate(params_in):
                    if p.requires_grad:
                        params_to_diff.append(p)
                        params_indices.append(i)
                
                if params_to_diff:
                    vjp_params = tr.autograd.grad(f_tensor, params_to_diff, grad_outputs=-lambd, allow_unused=True)
                    for i, g in zip(params_indices, vjp_params):
                        grad_params[i] = g
                
                return (None, *grad_params)

        params = _find_tensors(args)
        x_sol = Root.apply(x0, *params)
        
        # Full result
        def fun_np_meta(x_val):
             res = fun(rp.array(x_val, dtype=dt), *_replace_tensors(args, params, {'idx': 0}))
             return tr.as_tensor(res).detach().cpu().numpy()
        res = so.root(fun_np_meta, tr.as_tensor(x0).detach().cpu().numpy(), method=method, tol=tol, callback=callback, options=options)
        res = _convert_optimize_result(res)
        res.x = rp.array(x_sol, dtype=dt)
        return res
    
    else:
        return so.root(fun, x0, args=args, method=method, jac=jac, tol=tol, callback=callback, options=options)

def OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit): 
    if rp.use_jax: return jo.OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit)
    elif rp.use_torch: raise NotImplementedError('OptimizeResults not supported for Torch')
    else: return so.OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit)