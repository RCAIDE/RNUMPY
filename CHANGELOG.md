This file will be updated with new features, bug fixes, breaking changes, and other important information as versions are updated.

## 0.2.0 (2026-10-10)
First release since 0.1.0 (December 2024).

### Added
- PyTorch backend (`use_torch`), alongside NumPy and JAX, with autograd through `minimize` Jacobians.
- Differentiable `scipy.optimize.fsolve` (implicit-function gradients under JAX and torch), plus updated `scipy.optimize` functions.
- Write guard (`ensure_differentiable`): in-place writes on RNUMPY arrays raise an error with copy-and-paste `.at[].set()` / `rp.where` replacements.
- `rnp.at(x)`: the function form of `x.at[idx]` that also accepts plain `numpy.ndarray` (returns a modified copy of the same type) and torch tensors; JAX arrays use their native `.at`.
- `griddata`, `any`, `all`, `isclose`, `isfinite`, `pi`, `random.choice` and more NumPy functions; float64 controls (`enable_x64`).
- Tests (`tests/`) and CI on pull requests and pushes (NumPy on Ubuntu, macOS and Windows; JAX and torch on Ubuntu), with coverage uploaded to Codecov.

### Fixed
- `.at[].set()` under NumPy, argument order of `where`, `where` receiving `None`, and the `Rotation` class.
- Imports for JAX's `scipy.optimize`; interpolation, convex hull, B-spline and slicing issues under torch; default dtypes.
- Removed the repository-root `__init__.py` (invalid syntax; it stopped pytest from collecting the tests).

### Changed
- README: NumPy is the default backend; documented backend switching, `enable_x64()` and index updates.
