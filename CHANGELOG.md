This file will be updated with new features, bug fixes, breaking changes, and other important information as versions are updated.

## Unreleased
- Added `rnp.at(x)`: the function form of `x.at[idx]` that also accepts plain `numpy.ndarray` (returns a modified copy of the same type) and torch tensors; JAX arrays use their native `.at`.
- README: NumPy is the default backend; documented `enable_x64()` and index updates.
- Removed the repository-root `__init__.py` (invalid syntax; it stopped pytest from collecting the tests).
- `ensure_differentiable` is off by default, as in 0.1.0: in-place writes on RNUMPY arrays work unless the write guard is enabled.
- The 67 SciPy names that 0.1.0 exported at the top level (`rp.minimize`, `rp.erf`, `rp.lu_solve`, ...) are available at the top level again, as aliases of `rp.scipy.<name>`.
- Breaking: `rp.linalg` is the NumPy-style `linalg` (as `jnp.linalg`); in 0.1.0 it was `rp.scipy.linalg`. Use `rp.scipy.linalg` for `lu_factor`, `cho_solve`, etc. Nine `scipy.special` placeholders that only raised `NotImplementedError` in 0.1.0 were removed: `bernoilli`, `gammasgn`, `hyp1f1`, `kl_div`, `lpmn`, `lpmn_values`, `rel_entr`, `spence`, `sph_harm`.
