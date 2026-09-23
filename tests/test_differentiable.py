"""Regression tests for differentiable SciPy wrappers and autograd helpers."""

import importlib.util

import numpy as np
import pytest

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _reset_rp():
    import RNUMPY as rp
    rp.use_jax = False
    rp.use_torch = False
    rp.ensure_differentiable = True
    return rp


def _fsolve_problem(rp):
    def residual(z, p):
        z = rp.asarray(z)
        return rp.stack([z[0] ** 2 - p])

    def objective(p):
        z0 = rp.array([1.0])
        z = rp.scipy.fsolve(residual, z0, args=(p,))
        return z[0] ** 2

    return residual, objective


def _forbid_finite_diff(monkeypatch):
    import RNUMPY.scipy.optimize as opt

    def _boom(*_args, **_kwargs):
        raise AssertionError("finite-difference Jacobian used in an AD backend")

    monkeypatch.setattr(opt, "finite_diff_jacobian", _boom)


def _autograd_node_names(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.grad_fn is None:
        return []
    seen = set()
    names = []
    stack = [tensor.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        names.append(type(fn).__name__)
        stack.extend(nxt for nxt, _ in fn.next_functions)
    return names


def _outer_root_objective(rp):
    """Minimized when the inner root is 3, i.e. the outer variable is 9."""

    def objective(x):
        x = rp.asarray(x)

        def residual(z, x_inner):
            z = rp.asarray(z)
            x_inner = rp.asarray(x_inner)
            return rp.stack([z[0] ** 2 - x_inner[0]])

        z = rp.scipy.fsolve(residual, rp.asarray([1.0]), args=(x,))
        return (z[0] - 3.0) ** 2

    return objective


def test_numpy_finite_diff_grad():
    rp = _reset_rp()

    def f(x):
        return (x[0] - 1.0) ** 2

    g = rp.grad(f)(rp.array([0.5]))
    assert abs(float(g[0]) - (-1.0)) < 1e-4


def test_numpy_grad_through_fsolve():
    rp = _reset_rp()
    _, objective = _fsolve_problem(rp)
    g = rp.grad(objective)(rp.array(2.0))
    assert np.isfinite(float(g))


def test_numpy_fsolve_full_output_and_fprime():
    rp = _reset_rp()
    residual, _ = _fsolve_problem(rp)

    def fprime(z, p):
        z = rp.asarray(z)
        return rp.asarray([[2.0 * z[0]]])

    z, info, ier, mesg = rp.scipy.fsolve(residual, rp.array([1.0]), args=(2.0,), full_output=True)
    assert int(ier) == 1
    assert abs(float(z[0]) ** 2 - 2.0) < 1e-8
    assert "nfev" in info
    assert "njev" in info

    z_col = rp.scipy.fsolve(
        residual, rp.array([1.0]), args=(2.0,), fprime=fprime, col_deriv=1,
    )
    assert abs(float(z_col[0]) ** 2 - 2.0) < 1e-8


@needs_jax
def test_jax_fsolve_value_and_grad(monkeypatch):
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import RNUMPY as rp
    import jax.numpy as jnp

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = True
    rp.use_torch = False
    _, objective = _fsolve_problem(rp)
    p = jnp.array(2.0)
    v, g = rp.value_and_grad(objective)(p)
    assert np.isfinite(float(v))
    assert np.isfinite(float(g))


@needs_torch
def test_torch_fsolve_value_and_grad():
    import RNUMPY as rp
    import torch

    rp.use_jax = False
    rp.use_torch = True
    _, objective = _fsolve_problem(rp)
    p = torch.tensor(2.0, requires_grad=True)
    v, g = rp.value_and_grad(objective)(p)
    assert np.isfinite(float(v.detach()))
    assert abs(float(g) - 1.0) < 1e-4


@needs_jax
def test_jax_fsolve_does_not_finite_difference(monkeypatch):
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import RNUMPY as rp
    import jax.numpy as jnp

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = True
    rp.use_torch = False
    _, objective = _fsolve_problem(rp)
    v, g = rp.value_and_grad(objective)(jnp.array(2.0))
    assert abs(float(v) - 2.0) < 1e-4
    assert abs(float(g) - 1.0) < 1e-4


@needs_torch
def test_torch_fsolve_does_not_finite_difference(monkeypatch):
    import RNUMPY as rp
    import torch

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = False
    rp.use_torch = True
    _, objective = _fsolve_problem(rp)
    p = torch.tensor(2.0, requires_grad=True)
    v, g = rp.value_and_grad(objective)(p)
    assert abs(float(v.detach()) - 2.0) < 1e-4
    assert abs(float(g) - 1.0) < 1e-4


@needs_jax
def test_jax_fsolve_two_parameters(monkeypatch):
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import RNUMPY as rp
    import jax.numpy as jnp

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = True
    rp.use_torch = False

    def objective(a, b):
        def residual(z, a_inner, b_inner):
            z = rp.asarray(z)
            return rp.stack([a_inner * z[0] - b_inner])

        z = rp.scipy.fsolve(residual, rp.asarray([1.0]), args=(a, b))
        return z[0]

    ga, gb = rp.grad(objective, argnums=(0, 1))(jnp.array(2.0), jnp.array(6.0))
    assert abs(float(ga) - (-1.5)) < 1e-4
    assert abs(float(gb) - 0.5) < 1e-4


@needs_torch
def test_torch_fsolve_two_parameters(monkeypatch):
    import RNUMPY as rp
    import torch

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = False
    rp.use_torch = True

    def objective(a, b):
        def residual(z, a_inner, b_inner):
            z = rp.asarray(z)
            return rp.stack([a_inner * z[0] - b_inner])

        z = rp.scipy.fsolve(residual, rp.asarray([1.0]), args=(a, b))
        return z[0]

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(6.0, requires_grad=True)
    ga, gb = rp.grad(objective, argnums=(0, 1))(a, b)
    assert abs(float(ga) - (-1.5)) < 1e-4
    assert abs(float(gb) - 0.5) < 1e-4


@needs_jax
def test_jax_fsolve_inside_minimize(monkeypatch):
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import RNUMPY as rp
    import jax.numpy as jnp

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = True
    rp.use_torch = False
    objective = _outer_root_objective(rp)
    grad = rp.grad(objective)(jnp.asarray([4.0]))
    assert abs(float(grad[0]) - (-0.5)) < 1e-4

    result = rp.scipy.optimize.minimize(objective, jnp.asarray([4.0]), method="BFGS", tol=1e-10)
    assert abs(float(result.x[0]) - 9.0) < 1e-3


@needs_torch
def test_torch_fsolve_inside_minimize(monkeypatch):
    import RNUMPY as rp
    import torch

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = False
    rp.use_torch = True
    objective = _outer_root_objective(rp)
    graphs = []

    def wrapped(x):
        loss = objective(x)
        if isinstance(loss, torch.Tensor) and loss.grad_fn is not None:
            graphs.append(_autograd_node_names(loss))
        return loss

    x = torch.tensor([4.0], requires_grad=True)
    grad = rp.grad(wrapped)(x)
    assert abs(float(grad[0]) - (-0.5)) < 1e-4
    assert graphs
    for names in graphs:
        assert any("FSolveBackward" in name for name in names)
        assert sum("PowBackward" in name for name in names) <= 1
        assert len(names) < 16

    result = rp.scipy.optimize.minimize(wrapped, torch.tensor([4.0]), method="BFGS", tol=1e-10)
    assert abs(float(result.x[0]) - 9.0) < 1e-3
    assert graphs
    for names in graphs:
        assert any("FSolveBackward" in name for name in names)
        assert sum("PowBackward" in name for name in names) <= 1
        assert len(names) < 16


@needs_jax
def test_jax_fsolve_full_output(monkeypatch):
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import RNUMPY as rp
    import jax.numpy as jnp

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = True
    rp.use_torch = False
    residual, _ = _fsolve_problem(rp)
    z, info, ier, mesg = rp.scipy.fsolve(
        residual, rp.asarray([1.0]), args=(jnp.array(2.0),), full_output=True,
    )
    assert int(ier) == 1
    assert abs(float(z[0]) ** 2 - 2.0) < 1e-5
    assert int(info["njev"]) >= 1


@needs_torch
def test_torch_fsolve_full_output(monkeypatch):
    import RNUMPY as rp
    import torch

    _forbid_finite_diff(monkeypatch)
    rp.use_jax = False
    rp.use_torch = True
    residual, _ = _fsolve_problem(rp)
    p = torch.tensor(2.0, requires_grad=True)
    z, info, ier, mesg = rp.scipy.fsolve(
        residual, rp.asarray([1.0]), args=(p,), full_output=True,
    )
    assert int(ier) == 1
    assert abs(float(z[0].detach()) ** 2 - 2.0) < 1e-5
    assert int(info["njev"]) >= 1
    g = torch.autograd.grad(z[0] ** 2, p)[0]
    assert abs(float(g) - 1.0) < 1e-4


@needs_torch
def test_torch_ad_jacobian_rectangular():
    import RNUMPY as rp
    import torch
    from RNUMPY.scipy.optimize import _torch_ad_jacobian

    rp.use_jax = False
    rp.use_torch = True

    def residual(x, p):
        x = rp.asarray(x)
        return rp.stack([x[0] + x[1] - p, x[0] - x[1]])

    jac = _torch_ad_jacobian(residual, np.array([1.0, 2.0, 3.0]), (torch.tensor(1.0),), torch.float64, None)
    assert jac.shape == (2, 3)


def test_fsolve_exported_from_optimize():
    import RNUMPY as rp
    import RNUMPY.scipy.optimize as opt
    import RNUMPY.scipy.src as src_mod

    assert rp.scipy.fsolve is opt.fsolve
    assert not hasattr(src_mod, "fsolve")
