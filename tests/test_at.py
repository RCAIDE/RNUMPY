"""Tests for rp.at, the function form of x.at[idx].set(v) that accepts plain arrays."""

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


def test_at_numpy_matches_inplace_write():
    rp = _reset_rp()
    a = np.arange(12.0).reshape(4, 3)
    expected = a.copy()
    expected[:, 1] = [9.0, 8.0, 7.0, 6.0]
    b = rp.at(a)[:, 1].set(np.array([9.0, 8.0, 7.0, 6.0]))
    assert type(b) is np.ndarray
    assert np.array_equal(b, expected)
    assert np.array_equal(a, np.arange(12.0).reshape(4, 3))


def test_at_numpy_add():
    rp = _reset_rp()
    a = np.zeros((3, 2))
    b = rp.at(a)[0, 1].add(2.5)
    assert b[0, 1] == 2.5 and a[0, 1] == 0.0


def test_at_numpyarray_keeps_type_with_guard_on():
    rp = _reset_rp()
    a = rp.array([1.0, 2.0, 3.0])
    b = rp.at(a)[1].set(5.0)
    assert isinstance(b, type(a))
    assert np.array_equal(np.asarray(b), [1.0, 5.0, 3.0])


@needs_jax
def test_at_jax_is_native_and_differentiable():
    import jax
    import jax.numpy as jnp
    rp = _reset_rp()
    x = jnp.arange(4.0)
    assert np.array_equal(np.asarray(rp.at(x)[2].set(7.0)), np.asarray(x.at[2].set(7.0)))
    g = jax.grad(lambda p: rp.at(x)[1:3].set(p * 2.0).sum())(1.0)
    assert float(g) == 4.0


@needs_torch
def test_at_torch_returns_modified_clone():
    import torch
    rp = _reset_rp()
    x = torch.zeros(3)
    y = rp.at(x)[0].set(1.0)
    assert isinstance(y, torch.Tensor)
    assert float(y[0]) == 1.0 and float(x[0]) == 0.0


_METHODS = {
    "set":      lambda cur, v: v,
    "add":      lambda cur, v: cur + v,
    "subtract": lambda cur, v: cur - v,
    "multiply": lambda cur, v: cur * v,
    "divide":   lambda cur, v: cur / v,
    "power":    lambda cur, v: cur ** v,
    "min":      np.minimum,
    "max":      np.maximum,
}
_INDICES = {"slice": (slice(1, 3),), "column": (slice(None), 1), "entry": (2, 0)}


def _sample():
    return np.linspace(0.5, 2.0, 12).reshape(4, 3)


@pytest.mark.parametrize("index", list(_INDICES))
@pytest.mark.parametrize("method", list(_METHODS))
def test_at_numpy_methods_match_reference(method, index):
    rp = _reset_rp()
    x = _sample()
    idx = _INDICES[index]
    expected = x.copy()
    expected[idx] = _METHODS[method](expected[idx], 1.5)
    out = getattr(rp.at(x)[idx], method)(1.5)
    assert type(out) is np.ndarray
    assert np.array_equal(out, expected)
    assert np.array_equal(x, _sample())


@pytest.mark.parametrize("index", list(_INDICES))
def test_at_get(index):
    rp = _reset_rp()
    x = _sample()
    assert np.array_equal(rp.at(x)[_INDICES[index]].get(), x[_INDICES[index]])


@needs_jax
@pytest.mark.parametrize("index", list(_INDICES))
@pytest.mark.parametrize("method", list(_METHODS))
def test_at_numpy_matches_jax(method, index):
    import jax.numpy as jnp
    rp = _reset_rp()
    rp.enable_x64()
    x = _sample()
    idx = _INDICES[index]
    out_np = getattr(rp.at(x)[idx], method)(1.5)
    out_jax = getattr(rp.at(jnp.asarray(x))[idx], method)(1.5)
    np.testing.assert_allclose(out_np, np.asarray(out_jax), rtol=1e-14, atol=0)
